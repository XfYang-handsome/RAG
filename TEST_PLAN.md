# RAG 系统测试计划

> 本文件是 RAG 系统的量化评测执行计划，配套代码在 `eval/` 目录。
> 评测方法论、指标定义、配置细节见 `README.md` 第 16 章。

---

## 1. 测试目标

量化评估本 RAG 系统的**检索质量**与**生成质量**，回答两个核心问题：

| 关注点 | 核心问题 | 对应指标 |
|---|---|---|
| 召回精度 | 检索是不是把该找的都找出来了？有没有漏检/噪声？ | Context Recall、Context Precision |
| 推理准确率 | 模型有没有一本正经地胡说八道（幻觉）？ | Faithfulness |

## 2. 测试范围

### 2.1 评测集（20 题）

| 来源 | 题数 | 说明 |
|---|---|---|
| `README.md`（RAG 系统实现） | 5 | 单文档 |
| `rag.pdf`（RAG-AIGC 综述） | 5 | 单文档 |
| `2404.18231v2.pdf`（角色扮演 Agent 综述） | 5 | 单文档 |
| 跨文档 | 5 | 需综合多篇文档作答 |

生成方式：LLM 从文档 chunk 自动抽取 QA 对（`eval/generate_dataset.py`），输出到 `eval/data/dataset.json`。

### 2.2 组合矩阵（排列组合）

各维度对三种模式的作用范围：

| 维度 | 取值 | direct | rag | agentic |
|---|---|---|---|---|
| `mode` | direct / rag / agentic | — | — | — |
| `retrieval_mode` | vector / hybrid / tree | ✗ | ✓ | ✓ |
| `rewrite`（查询改写） | on / off | ✗ | ✓ | ✗ |
| `tool_calling`（工具调用） | on / off | ✗ | ✓ | ✗ |
| `websearch`（联网） | on / off | ✗ | ✓ | ✓ |

**组合总数 = 1（direct）+ 24（rag）+ 6（agentic）= 31 种**。

### 2.3 本期执行子集：检索模式对比（6 组）

聚焦最核心的「检索模式对召回/精度的影响」，纯净对比（关闭 rewrite/tool/websearch 避免干扰归因）：

```
rag_vector_rw0_tc0_ws0      rag_hybrid_rw0_tc0_ws0      rag_tree_rw0_tc0_ws0
agentic_vector_ws0          agentic_hybrid_ws0          agentic_tree_ws0
```

### 2.4 指标体系（5 指标）

| 指标 | 需要 ground_truth | 对应问题 |
|---|---|---|
| Faithfulness（忠实度） | ❌ | 幻觉 |
| Context Recall（上下文召回率） | ✅ | 漏检 |
| Context Precision（上下文精度） | ✅ | 噪声/排序 |
| Answer Relevancy（答案相关性） | ❌ | 答非所问 |
| Answer Correctness（正确性） | ✅ | 端到端对错 |

## 3. 测试环境

| 项 | 配置 |
|---|---|
| 生成 LLM | `deepseek-ai/DeepSeek-V4-Pro-0813`（SiliconFlow 国际站 `api.siliconflow.com`） |
| judge LLM | `deepseek-ai/DeepSeek-V3.2`（非 reasoning，保证结构化输出稳定） |
| Embedding | `Qwen/Qwen3-Embedding-0.6B`（已重新入库 994 条 chunk） |
| Reranker | `bge-reranker-v2-m3`（本地） |
| 评测框架 | Ragas 0.4.3（独立 venv：`eval/.venv`） |

> 密钥敏感信息在 `config/models.json`（已 gitignore，不入库）。

## 4. 执行步骤

```powershell
# ① 生成评测集（20 题）
poetry run python eval/generate_dataset.py

# ② 采集：跑 pipeline，采集 answer + contexts（断点续跑，每题落盘）
poetry run python eval/collect.py --subset mode-comparison

# ③ 评分：Ragas 5 指标（eval 独立 venv）
eval/.venv/Scripts/python eval/score.py --subset mode-comparison

# ④ 汇总报告
poetry run python eval/report.py
```

## 5. 产出物

| 产出 | 路径 | 说明 |
|---|---|---|
| 评测集 | `eval/data/dataset.json` | 20 题 QA 对 |
| 采集结果 | `eval/data/runs/{combo_id}.json` | answer + contexts |
| 评分结果 | `eval/data/scores/{combo_id}.json` | 每指标 per-question + avg |
| 最终报告 | `eval/data/report.md` | 组合 × 指标矩阵 + 维度对比 |

## 6. 时间预估

| 阶段 | 规模 | 预估 |
|---|---|---|
| 采集 | 6 组合 × 20 题 = 120 次 | 约 1.5~2.5 小时 |
| 评分 | 120 题次 × 5 指标 | 约 6 小时 |
| **合计** | — | **约 8 小时** |

## 7. 当前进度

| 步骤 | 状态 |
|---|---|
| 环境搭建（ragas 独立 venv） | ✅ 完成 |
| 配置切换 + 重新入库（Qwen3） | ✅ 完成（994 条） |
| 评测集生成 | ✅ 完成（20 题） |
| 端到端冒烟（rag_vector 前 2 题） | ✅ 通过（5 指标全出分） |
| 正式采集 + 评分 + 报告 | ⏸️ 待执行 |

## 8. 风险与注意事项

1. **换 embedding 必须重新入库**：换模型（即使维度相同）会导致向量空间不兼容，检索失效。已用 Qwen3 重新入库。
2. **judge 必须用非 reasoning 模型**：reasoning 模型的 thinking 无法禁用，会导致 faithfulness/correctness 结构化输出超时。
3. **两个评分坑已修复**：`max_tokens` 需 ≥4096（防 JSON 截断）、`timeout` 需 ≥600s（防大 JSON 超时）。
4. **成本敏感**：全量 31 组合成本高，故本期只跑 6 组子集。扩展时注意 API 额度。
5. **指标组合看**：只优化 Recall 会拉大 top_k → Precision 崩 → 噪声多 → Faithfulness 降，需三者平衡。
