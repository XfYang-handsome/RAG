# RAG 知识库系统 · 使用与实现详解

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Vue 3](https://img.shields.io/badge/Vue-3-4FC08D.svg?logo=vuedotjs&logoColor=white)](https://vuejs.org/)
[![Milvus](https://img.shields.io/badge/Milvus-2.x-00A1EA.svg)](https://milvus.io/)

> 一个**生产级检索增强生成（RAG）知识库系统**：把私有文档智能解析入库，结合向量检索、文档树导航与 Agentic 多步推理，让大模型基于你的文档准确作答（带出处、不瞎编）。

## 核心特性

- **双引擎文档解析**：`chunklet-py` 语义分块 + RAGFlow `deepdoc` 版面/表格/OCR，自动构建「章节 → 小节 → 段落」文档树
- **多路检索 + 重排序**：向量 / 混合 / 树导航三种检索，配合 BGE、Jina Reranker 重排序
- **三种对话模式**：`direct`（纯 LLM）、`rag`（固定检索链路）、`agentic`（多步检索循环）
- **MCP 集成**：内置 MCP 工具管理，可接入外部工具扩展问答能力
- **评测闭环**：基于 Ragas 自动生成 QA 评测集，量化检索与生成质量
- **异步入库 + 一键部署**：Celery + Redis 任务队列，Windows / Linux / macOS 一键脚本

---

> 这是一份**面向所有人的教程式文档**。无论你是第一次接触 RAG，还是想深入理解本项目的实现，都能从零看懂。
>
> 每个功能都按「**是什么 → 为什么 → 怎么做 → 代码在哪 → 核心逻辑**」的结构讲解。

---

## 目录

- [1. 这个系统是干什么的](#1-这个系统是干什么的)
- [2. 整体架构：两条主干](#2-整体架构两条主干)
- [3. 环境要求与安装](#3-环境要求与安装)
- [4. 核心概念速览（先建立心智模型）](#4-核心概念速览先建立心智模型)
- [5. 数据入库：把文档「装」进系统](#5-数据入库把文档装进系统)
- [6. 存储层：SQLite 树库 + Milvus 向量库](#6-存储层sqlite-树库--milvus-向量库)
- [7. 检索：把答案「找」出来](#7-检索把答案找出来)
- [8. Agentic RAG（多步检索循环）](#8-agentic-rag多步检索循环)
- [9. Reranker 重排序](#9-reranker-重排序)
- [10. LLM 封装与模型工厂](#10-llm-封装与模型工厂)
- [11. MCP 集成与工具管理](#11-mcp-集成与工具管理)
- [12. 前端页面](#12-前端页面)
- [13. 配置体系](#13-配置体系)
- [14. API 接口](#14-api-接口)
- [15. 故障排查](#15-故障排查)
- [16. RAG 评测（Ragas）](#16-rag-评测ragas)
- [17. DeepSeek Harness 融合](#17-deepseek-harness-融合)
- [附录：项目文件速查表](#附录项目文件速查表)

---

## 1. 这个系统是干什么的

### 1.1 一句话说明

这是一个**「让大模型能回答你私有文档内容」的系统**（RAG，Retrieval-Augmented Generation，检索增强生成）。

大模型（LLM）虽然知识渊博，但它**不知道你的私有文档**——比如你的公司手册、技术规范、研究论文。RAG 的思路是：

```
你上传文档 → 系统把文档"拆碎、向量化、存起来"
        ↓
你提问     → 系统先"检索"出最相关的片段
        ↓
        → 把这些片段连同问题一起交给大模型
        ↓
大模型     → 基于片段生成答案（带出处，不瞎编）
```

### 1.2 一个类比：图书馆

把整个系统想象成一座**图书馆**：

| 图书馆 | 本系统 |
|---|---|
| 书 | 你上传的文档 |
| 书的目录 | 文档树（章节/小节层级） |
| 图书管理员 | 检索器（Retriever） |
| 读者借书时"查目录找书" | 检索（找到最相关的片段） |
| 读者读完写总结 | 大模型生成答案 |

### 1.3 一个问题的完整旅程

以「角色型 Agent 有什么人格？它们的区别是什么？」为例：

```
① 你提问
     ↓
② 复杂度判断：COMPLEX（需要拆解）
     ↓
③ 拆解成两个信息需求：R1=人格类型，R2=各人格区别
     ↓
④ 检索：在"角色型 Agent 综述"这篇文档的树里，逐层找到相关章节
     ↓
⑤ 找到若干"证据"片段（正文段落）
     ↓
⑥ 大模型基于这些证据，生成带引用的答案
     ↓
⑦ 前端流式展示（Markdown 排版 + 引用来源）
```

---

## 2. 整体架构：两条主干

系统有两条完全独立、但最终交汇的主干：**入库**（把文档装进去）和**问答**（把答案找出来）。

### 2.1 入库：两种解析方式

上传文件时，由「增强解析」开关决定走哪条路（**默认开启**）：

| | 普通解析（enhance=false） | 增强解析（enhance=true，默认） |
|---|---|---|
| 解析引擎 | chunklet-py（语义分块） | RAGFlow deepdoc（版面/表格/OCR） |
| 产出结构 | 无层级，纯"父子块" | 文档树（章节→小节→段落/表格/图） |
| 存储 | Milvus 父子双 Collection | SQLite 存树 + Milvus 存 chunk |
| 检索能力 | 向量/混合检索 | 向量/混合/**树导航**检索 |

### 2.2 问答：三种模式

前端「对话模式」三选一：

| 模式 | 说明 | 核心文件 |
|---|---|---|
| **direct** | 纯 LLM 推理，不检索知识库 | `server.py` |
| **rag** | LangGraph 固定检索链路（检索→重排→评估→生成） | `rag_graph.py` |
| **agentic** | 多步检索循环（拆解→循环→评估→合成） | `agentic_rag/` |

### 2.3 架构全景图

```
                             ┌─────────────────────────────┐
                             │  前端（Vue3 SPA，frontend/）  │
                             │  构建产物 static/dist/        │
                             └──────────────┬──────────────┘
                                            │ HTTP / SSE
                             ┌──────────────▼──────────────┐
                             │     server.py（FastAPI）     │
                             │  /chat /upload /config ...   │
                             └──┬───────────┬───────────┬──┘
                   问答          │           │           │  上传入库（异步）
        ┌────────────────────────┘           │           └──────────────┐
        ▼                                    ▼                          ▼
┌───────────────┐                  ┌──────────────────┐        ┌──────────────┐
│   rag_graph   │                  │   agentic_rag    │        │ Redis(broker)│
│   固定链路    │                  │   多步循环       │        └──────┬───────┘
└───────┬───────┘                  └────────┬─────────┘               │
        │                                   │                  ┌──────▼───────┐
        └───────────────┬───────────────────┘                  │ Celery Worker│
                        │                                      │ 解析/切分/    │
              ┌─────────▼─────────┐   ┌───────────┐            │ 向量化/入库   │
              │  tree_retrieval   │   │ db_service │◄───────────┤ (ingest_file)│
              │  纯树导航检索     │   │ 向量/混合  │            └──────┬───────┘
              └─────────┬─────────┘   └─────┬─────┘                   │
                        │                   │                          │
              ┌─────────▼─────────┐   ┌─────▼─────┐            ┌──────▼───────┐
              │ tree_store(SQLite)│   │milvus_store│            │ ingest_queue │
              │  文档树+原文      │   │  向量库    │            │ (SQLite任务表)│
              └───────────────────┘   └───────────┘            └──────────────┘
```

**关键设计**：SQLite 只存「结构 + 原文」，Milvus 只存「向量 + chunk」。二者通过 `node_id` / `parent_node_id` 关联，**分工明确、互不越界**。

**入库异步化**：`/upload` 只负责「落盘 + 提交任务 + 立即返回 `task_id`」，解析/切分/向量化/入库全部在独立的 **Celery Worker** 进程中执行（经 Redis 消息队列分派），主进程绝不阻塞。任务状态由 `ingest_queue` 的 SQLite 任务表记录，前端轮询查询。

**入库步骤化（Phase 3）**：Worker 内的入库拆成 `parse → chunk → embed → index` 四个步骤任务，每步产物落盘到 `data/steps/{task_id}/`，支持**步骤级重跑**（某步失败只重跑该步，前置产物复用）。可选**双队列**（`ingest.dual_queue=true`）：parse/chunk/index 进 `parse_queue`（CPU 密集）、embed 进 `embedding_queue`（网络 IO 密集），两个 Worker 独立扩容。

### 2.4 项目结构

```
RAG/
├── __main__.py              # 主程序入口（--mcp 同步启停 MCP；--celery 同步启停 Celery Worker）
├── server.py                # FastAPI HTTP 服务 + 所有 API + 挂载 MCP 管理路由
├── rag_graph.py             # LangGraph 固定检索链路（retrieve/rerank/grade/kb_retry/generate）
├── tree_retrieval.py        # 纯树导航检索（通用模块：tree_navigate + 文档路由 + 三级降级，输出统一 dict）
├── db_service.py            # 数据服务（embedding + Milvus 检索/入库/管理，多库切换；含 parse/chunk/embed/index 四步函数）
├── celery_app.py            # Celery 应用 + 入库任务（Phase 3 四步链式任务 + 双队列路由）
├── ingest_queue.py          # 异步入库任务队列（SQLite 任务表 + 状态机 + 步骤产物落盘 + 步骤级重跑）
├── chat_history.py          # 对话历史存储（SQLite，data/chat_history.db，与知识库隔离）
├── structure_resolver.py    # 结构归位（deepdoc 扁平元素 → 文档树）+ TreeNode 序列化/反序列化
├── chunk_builder.py         # 结构树 → Retrieval Chunk 切分
├── tree_store.py            # 文档树持久化（SQLite）+ 章节路径恢复 + 结构查询原语
├── summarizer.py            # 章节摘要 + 文档主旨摘要（LLM）
├── dsml_read.py             # DSML 工具调用解析器（DeepSeek V4 文本兼容层）
├── embedding.py             # Embedding + chunklet-py 父子块切分
├── llm.py                   # LLM 封装（ChatOpenAI / DoubaoLLM）
├── llm_factory.py           # 模型工厂（按 kind 取模型，get_model）
├── milvus_store.py          # Milvus 封装（父子双 Collection + 混合检索）
├── reranker.py              # Reranker（本地 HF 推理 / 远程 API）
├── config_loader.py         # 系统配置加载（config/config.json）
├── store_config.py          # 模型/数据库配置管理（models.json、db.json）
├── deepdoc/                 # RAGFlow deepdoc 解析引擎
├── deepdoc_extractor.py     # deepdoc 增强解析封装
├── common/                  # 通用工具（text_utils.py：健壮 JSON 解析/CJK/翻译）
├── rag/                     # RAGFlow rag 轻量替代实现
├── agentic_rag/             # Agentic RAG 子包（多步检索循环）
│   ├── state.py             # 显式状态（AgentState）
│   ├── router.py            # 复杂度路由（SIMPLE/MEDIUM/COMPLEX）
│   ├── planner.py           # 需求拆解
│   ├── controller.py        # 决策器（Gap→Action）
│   ├── executor.py          # Action 执行器
│   ├── evaluator.py         # 证据评估
│   ├── stopping.py          # 停止策略
│   ├── synthesizer.py       # 带引用合成
│   ├── retriever.py         # 检索抽象 + Router + mode_to_tool
│   ├── agent.py             # 编排层
│   └── settings.py          # 配置读取（agentic.* 前缀）
├── mcp_service/             # MCP 服务子包（工具全部注册于此）
│   ├── __init__.py          # 包初始化
│   ├── __main__.py          # MCP 服务器入口（注册 6 个工具，含启用检查）
│   ├── manager.py           # MCP 服务器管理器（CRUD + 生命周期 + 工具 + 日志 + 缓存）
│   ├── tool_bridge.py       # 主程序调用 MCP 工具的唯一桥接（动态拉取工具清单 + 单工具调用）
│   ├── websearch.py         # 联网搜索（Google News RSS → Bing → 百度）
│   ├── math_tools.py        # 数学计算核心（π / LaTeX 算式）
│   └── knowledge_tools.py   # 知识库检索工具封装
├── frontend/                 # 前端源码（Vue3 + Naive UI + Pinia + Vite）
│   ├── package.json          # 前端依赖与脚本（dev/build）
│   ├── vite.config.js        # Vite 配置（开发代理到 8000；构建产物输出到 static/dist/）
│   ├── index.html            # SPA 入口
│   └── src/
│       ├── main.js           # 应用入口（挂载 Pinia + Router）
│       ├── App.vue           # 根组件（NConfigProvider 主题 + 路由视图）
│       ├── router/           # 路由（/#/ 主界面、/#/mcp MCP 管理）
│       ├── stores/           # Pinia 状态（theme/chat/settings/upload/mcp）
│       ├── api/              # 后端接口封装（http.js / sse.js / index.js）
│       ├── utils/markdown.js # Markdown + KaTeX 渲染（支持 4 种 LaTeX 分隔符）
│       ├── theme/            # Naive UI 主题覆盖（亮/暗）
│       ├── styles/global.css # 设计令牌（CSS 变量，亮暗主题）
│       ├── views/            # ChatView.vue（主界面）/ McpView.vue（MCP 管理）
│       └── components/       # chat/（消息/思考/工作台/引用）+ panel/（右侧面板）+ mcp/
├── eval/                     # RAG 评测框架（Ragas，独立于主程序，见第 16 章）
│   ├── config.py             # 组合矩阵 + 指标清单 + judge 模型配置
│   ├── generate_dataset.py   # 评测集生成（LLM 生成 QA 对）
│   ├── collect.py            # 组合遍历采集（answer + contexts）
│   ├── score.py              # Ragas 评分（5 指标，独立 venv 运行）
│   ├── report.py             # 汇总报告
│   ├── run_full.py           # 一键编排（生成→采集→评分→报告，自动跨环境 + 断点续跑）
│   └── data/                 # dataset.json / runs/ / scores/ / report.md
├── config/
│   ├── config.json          # 系统配置
│   ├── models.json          # 模型配置（llm/tool_llm/summary/embedding/reranker）
│   ├── mcp_servers.json     # MCP 服务器列表
│   └── db.json              # 数据库配置
├── static/
│   └── dist/                # 前端构建产物（index.html + assets/，由 npm run build 生成）
├── pyproject.toml           # Poetry 配置（torch cu128 + onnxruntime-gpu 1.26，CUDA 12.8 对齐）
├── poetry.lock
└── README.md
```

---

## 3. 环境要求与安装

### 3.0 一键部署（全新电脑推荐）

把仓库拷贝到新电脑后，只需两步即可跑起来。脚本会自动完成：检测 GPU 选依赖版本 → 生成配置 → `poetry install` → 下载 deepdoc 模型 → 构建前端 → 用 `deploy/docker-compose.yml` 拉起 Milvus + Redis → 健康检查。

> **第一步：装好 4 个系统依赖**（一次性）
> | 依赖 | 用途 | Windows | Linux | macOS |
> |------|------|---------|-------|-------|
> | Python 3.12 | 运行后端 | [python.org](https://www.python.org/downloads/) | `apt install python3.12` | `brew install python@3.12` |
> | Poetry | 装 Python 依赖 | `(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content \| py -` | `curl -sSL https://install.python-poetry.org \| python3 -` | `brew install poetry` |
> | Docker | 跑 Milvus/Redis | 装 [Docker Desktop](https://www.docker.com/products/docker-desktop/) | `curl -fsSL https://get.docker.com \| sh` | 装 Docker Desktop |
> | Node.js(+npm) | 构建前端 | [nodejs.org](https://nodejs.org/) LTS | `curl -fsSL https://deb.nodesource.com/setup_22.x \| sudo -E bash - && sudo apt install -y nodejs` | `brew install node` |
>
> 缺什么脚本会提示并按系统给出对应命令。NVIDIA GPU 可检测到并自动用 `cu128`，无 GPU 则自动切 CPU 版依赖（`--cpu`/`--gpu` 可强制指定）。

> **第二步：跑初始化脚本**
> ```bash
> # Windows：双击 deploy/setup.bat（或）
> python deploy\setup.py
>
> # Linux / macOS：
> bash deploy/setup.sh   # 等价于 python3 deploy/setup.py
> ```
> 脚本按顺序执行，中途会 `poetry install`（首次较久）并下载模型，耐心等它跑完。

> **第三步：填 API Key 并启动**
> 1. 编辑 `config/models.json`（或启动后在 Web「系统设置」里填）—— 补上 LLM / Embedding / Reranker 的 API Key（如硅基流动）。
> 2. 启动：
>    - Windows：双击 `deploy/start.bat`
>    - Linux / macOS：`bash deploy/start.sh`
>    - 等价于 `poetry run python __main__.py --mcp --celery`
> 3. 浏览器打开 http://127.0.0.1:8000

**脚本清单与参数**：

| 脚本（均在 `deploy/` 目录） | 作用 |
|------|------|
| `deploy/setup.bat` / `deploy/setup.sh` | 一键初始化（首次必跑一次） |
| `deploy/start.bat` / `deploy/start.sh` | 一键启动（主程序 + MCP + Celery Worker） |
| `deploy/stop.bat` / `deploy/stop.sh` | 一键停止 Milvus + Redis 容器 |

`deploy/setup.py` 常用参数：`--cpu`（强制 CPU 依赖）、`--gpu`（强制 cu128）、`--restore`（把被 `--cpu` 改写的 `pyproject.toml` 恢复为 GPU 版）、`--skip-deps/--skip-models/--skip-frontend/--skip-services`（跳过已完成的步骤）。

> **数据与迁移**：Milvus / Redis / deepdoc 模型数据分别落在项目目录下的 `./milvus`、`./redis`、`./models`（均为 bind mount，已被 `.gitignore` 忽略）。**拷贝整个项目目录即可整体带走数据**，新机上跑完 `setup` 后数据直接可用。切换容器编排与手动 `docker run`（见 3.3/3.4）二者等价，任选其一。

> **首次启动说明**：`deploy/docker-compose.yml` 首次拉起 Milvus 需拉镜像并初始化（约 1~2 分钟），`deploy/setup.py` 会做健康检查并提示结果，超时属正常，稍后重跑或看容器日志即可。

### 3.1 环境

- **Python**：3.12
- **包管理**：Poetry
- **Milvus**：Docker standalone（本地）或远程服务
- **Redis**：Docker（异步入库任务队列的消息中间件）
- **GPU**（可选但推荐）：NVIDIA GPU（RTX 50 系列需 cu128 版 torch + onnxruntime-gpu 1.26，见下）

### 3.2 安装步骤

```bash
# 1. 安装依赖
poetry install

# 2. 验证 GPU（reranker 用 torch，deepdoc 用 onnxruntime-gpu）
poetry run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望输出：2.7.1+cu128 True
poetry run python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
# 期望 providers 含 CUDAExecutionProvider
```

> **GPU 加速说明**：本项目有两套推理框架，**CUDA 版本必须对齐**：
> - `torch`（cu128）→ 用于 reranker，基于 CUDA 12.8
> - `onnxruntime-gpu 1.26` → 用于 deepdoc 解析（OCR/版面/表格识别），同样基于 CUDA 12.8
>
> 二者复用同一套 CUDA 12.8 DLL，互不冲突。**不要升级到 `onnxruntime-gpu 1.27+`**：1.27 起切换为 CUDA 13，会与 torch 的 cu128 冲突（`import torch` 报 `cudnn_cnn64_9.dll` 加载失败）。这也是 `pyproject.toml` 里锁定 `<1.27.0` 的原因。
>
> deepdoc 四个模型（det/layout/rec/tsr）走 GPU 后，大 PDF 解析实测提速约 2.2 倍（50 页论文 330s → 147s）。

### 3.3 启动 Milvus

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  -e ETCD_USE_EMBED=true -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
  -e COMMON_STORAGETYPE=local \
  -v milvus_data:/var/lib/milvus \
  milvusdb/milvus:v2.4.15
```

### 3.4 启动 Redis

```bash
docker run -d --name rag-redis -p 6379:6379 redis:7-alpine
```

### 3.5 启动 Celery Worker

异步入库需要独立的 Worker 进程（解析/切分/向量化/入库全在这里执行，与主服务进程隔离）。

**方式一（推荐）：随主程序一键拉起**——`__main__.py` 加 `--celery` 开关，Worker 作为子进程随主程序启停，一条命令搞定：

```bash
poetry run python __main__.py --celery       # 只起主服务 + Worker
poetry run python __main__.py --mcp --celery # 主服务 + MCP + Worker 一起起
```

Worker 日志写入 `data/logs/celery_worker.log`，Ctrl+C 退出时 Worker 随主程序一起关闭（杀整棵进程树，不残留孤儿进程，见 11.6）。

**中断与恢复**：关闭程序时，`__main__.py` 会把「运行中」的入库任务标记为「失败：服务已关闭」，避免其状态永久卡在中间态；下次启动时，`server.py` 会自动把上次遗留的「运行中」任务统一标记为「失败：服务重启，任务中断（可重试）」（无论上次是正常关闭、强杀还是崩溃）。这些中断任务会在前端的「最近入库任务」列表里显示为「失败」，可一键重试（步骤级重跑，从失败步骤继续）。

**方式二：独立终端单独起 Worker**（需要更细粒度的队列/并发控制时）：

```bash
# 单队列（默认 ingest.dual_queue=false）：一个 Worker 消费所有步骤
# Windows 不支持 prefork，用 --pool=solo（单进程）；IO 密集任务也可用 --pool=threads
poetry run celery -A celery_app worker --loglevel=info --pool=solo

# 并发处理多文件：线程池（适合纯文本/在线 embedding 为主）
poetry run celery -A celery_app worker --loglevel=info --pool=threads --concurrency=4

# 双队列（config.json 设 ingest.dual_queue=true 后）：两个 Worker 独立扩容
poetry run celery -A celery_app worker -Q parse_queue --loglevel=info --pool=solo
poetry run celery -A celery_app worker -Q embedding_queue --loglevel=info --pool=threads --concurrency=4
```

> **为什么 Worker 要单独起？** 主服务进程只负责「提交任务 + 查状态」，绝不在 HTTP 请求内执行解析和向量化（可能耗时几分钟）。Worker 独立进程崩溃/重启不影响主服务，且任务会经 Redis 自动恢复重投。`--celery` 开关本质就是帮你在后台拉起这个独立 Worker 进程，并保证随主程序一起退出。
>
> **为什么上传文件显示「排队中」？** 那是 Worker 还没启动（或正在跑别的文件）导致的 `PENDING` 状态——任务已投进 Redis 队列，但没人消费。确认 Worker 已启动即可。
>
> **并发与池的选择**：deepdoc PDF 解析是 CPU 密集（受 Python GIL 限制，线程并发无收益甚至线程不安全），在线 embedding 是网络 IO 密集（线程并发收益明显）。所以：多 PDF → 起多个 `--pool=solo` Worker 进程（进程级并发）；以纯文本/在线 embedding 为主 → 用 `--pool=threads --concurrency=4`；任务量大 → 开双队列按类型分池。
>
> **单队列 vs 双队列**：任务量小时用单队列（简单，一个 Worker 全跑）；任务量上来、embedding 排队拖慢解析时，开双队列把 CPU 密集（parse/chunk/index）和网络 IO 密集（embed）分流，各自独立扩容/限流。切换只需改 `config.json` 的 `ingest.dual_queue` + 改 Worker 启动参数，**无需改任何代码**。

### 3.6 构建前端（首次运行或改前端后必做）

前端是 Vue3 工程，源码在 `frontend/`，运行时加载的是**构建产物** `static/dist/`。首次运行前必须先构建一次：

```bash
# 1. 安装前端依赖（首次）
cd frontend
npm install

# 2. 构建（产物输出到 static/dist/，server.py 直接读它）
npm run build
```

> **开发模式**：改前端时不想每次手动 build，可开 Vite 开发服务器（热更新）：
> ```bash
> cd frontend
> npm run dev    # 起在 5173，已配置代理到后端 8000（含 SSE 透传）
> ```
> 浏览器打开 `http://localhost:5173`，改动即时生效。改完再 `npm run build` 更新生产产物。

> **注意**：`server.py` 会缓存 `static/dist/index.html` 内容，改完前端重新 build 后需**重启主程序**，否则加载的还是旧产物。

### 3.7 启动主程序

```bash
# 推荐：主程序 + MCP 服务器一起启动（工具可用）
poetry run python __main__.py --mcp

# 或只启动主程序（知识库问答可用，工具不可用）
poetry run python __main__.py

# 需要异步入库时，再加 --celery 同步拉起 Worker（见 3.5）
poetry run python __main__.py --mcp --celery
```

浏览器打开 `http://127.0.0.1:8000`。

> **为什么 torch 要装 cu128？** PyPI 上的 `torch` 默认是 CPU 版（版本号带 `+cpu` 后缀）。RTX 50 系列（Blackwell 架构）需要 CUDA 12.8+，只有 PyTorch 官方源的 `cu128` wheel 支持。本项目已在 `pyproject.toml` 里显式声明了 cu128 源。

---

## 4. 核心概念速览（先建立心智模型）

这一节是**零基础必读**。每个概念用一句话 + 一个类比讲清楚，后面的章节会反复用到。

### 4.1 向量（Embedding）

**是什么**：把一段文字转换成一组数字（比如 1024 个浮点数），让"语义相近的文字在数学上距离也近"。

**类比**：给每句话发一个"GPS 坐标"，意思相近的话坐标也靠近。

**代码在哪**：`embedding.py` 的 `ChatOpenAIEmbeddingWrapper`

```python
text = "什么是向量检索"
vector = embedding.embed_text(text)   # 得到 [0.12, -0.33, ...] 共 N 维
```

### 4.2 Chunk（分块）

**是什么**：把一篇长文档切成若干小段，每段叫一个 chunk，是检索的最小单元。

**为什么**：大模型一次能处理的字数有限，且切小块后"检索命中更精准"（不会因为一段里混了无关内容而命中不准）。

**类比**：把一本书拆成"一页页"，检索时找"最相关的那几页"而不是整本书。

### 4.3 父子块（Parent-Child Chunk）

**是什么**：一种"双层切分"策略——**子块**（小，128~256 tokens）用来精准检索；**父块**（大，512~1024 tokens）用来给大模型看完整上下文。

**为什么**：子块太小会丢失上下文，父块太大检索不准。父子块结合：**用小的找，用大的答**。

```
父块（大，给 LLM 看）  ┌─────────────────────────────┐
                       │  ┌──────┐ ┌──────┐ ┌──────┐  │
子块（小，用来检索）     │  │ 子块0 │ │ 子块1 │ │ 子块2 │  │
                       │  └──────┘ └──────┘ └──────┘  │
                       └─────────────────────────────┘
```

**代码在哪**：`embedding.py` 的 `ParentChildChunker`

### 4.4 文档树（Document Tree）

**是什么**：把文档的结构（章节层级）还原成一棵树。

```
Document（文档根）
└── Section（第一章）
    ├── Paragraph（段落）
    ├── Table（表格）
    └── Section（1.1 节）
        └── Paragraph（段落）
```

**为什么**：有了树，检索就能"顺着目录找"，而不是在大海捞针。

**代码在哪**：`structure_resolver.py`（建树）、`tree_store.py`（存树）

### 4.5 Rerank（重排序）

**是什么**：先用"快但粗"的方式召回一批候选（如向量检索 top 20），再用"慢但准"的模型精排（rerank），留下最相关的 top 5。

**类比**：招聘先筛简历（粗筛 100 份），再面试精挑（留下 5 人）。

**代码在哪**：`reranker.py`

### 4.6 BM25 与 RRF（混合检索的两件套）

- **BM25**：经典的关键词检索算法，擅长"精确词匹配"（比如搜"幽境危战"一定要命中这个词）。
- **RRF（Reciprocal Rank Fusion）**：把"向量语义检索的结果"和"BM25 关键词检索的结果"融合排序，取长补短。

**为什么混合**：向量检索擅长"意思相近但措辞不同"，BM25 擅长"精确关键词"。两者结合，召回更全。

**代码在哪**：`milvus_store.py` 的 `search_hybrid`

### 4.7 MCP（Model Context Protocol）

**是什么**：一个"让大模型调用外部工具"的开放协议。本项目的**所有工具**（联网搜索、数学计算、知识库检索）都注册在 MCP 服务器上。

**为什么**：主程序"不持有"工具，需要时通过 MCP 协议调用——职责清晰，且能被外部模型复用。

**代码在哪**：`mcp_service/`

---

## 5. 数据入库：把文档「装」进系统

入库有两条路径。无论哪条，都是**异步执行**：上传接口立即返回 `task_id`，真正的解析/切分/向量化/入库在 Celery Worker 里跑。

### 5.0 为什么异步？任务怎么流转

**是什么**：上传文件后，接口**立刻返回**，不等待解析完成；前端轮询任务状态，看到进度。

**为什么**：解析（deepdoc PDF/OCR）+ 向量化（在线 API）可能耗时几分钟，若在 HTTP 请求内同步执行，会长时间阻塞接口、占满并发。异步后接口只需几十毫秒。

**怎么做（任务状态机）**：

```
PENDING → PARSING → CHUNKING → EMBEDDING → INDEXING → DONE
                ↘                              ↘
                 └────────── FAILED ←──────────┘
```

**代码在哪**：`ingest_queue.py`（任务表 + 状态机 + 分派 + 步骤产物落盘）、`celery_app.py`（Celery 步骤任务）、`db_service.py` 的 `parse_step`/`chunk_step`/`embed_step`/`index_step`（四步纯函数）。

**核心逻辑（Phase 3 步骤化链式）**：

```python
# server.py（上传接口，立即返回）
task_id = ingest_queue.submit(filename, enhance, save_path)  # 落盘 + 创建任务 + 分派第一步

# ingest_queue._dispatch（按产物落盘情况决定起点，支持步骤级重跑）
step = _current_step(task_id)  # 无 parse 产物→parse；有 parse 无 chunk→chunk；...；有 embed→index

# celery_app.py（Worker 进程执行，四步链式：每步完成落盘产物再 delay 下一步）
@celery_app.task(bind=True, name="rag.parse_document", max_retries=3)
def parse_document(self, task_id):
    if not has_step(task_id, "parse"):          # 幂等：产物已在则跳过
        pr = parse_step(file_path, source, enhance, on_progress=...)
        save_step(task_id, "parse", pr)          # 产物落盘 data/steps/{task_id}/parse_result.json
    chunk_document.delay(task_id)                # 下一步
# chunk_document / embed_document / index_document 同理；index 完成后标记 DONE 并清理产物+文件
```

**关键设计**：
- **步骤级重跑（幂等）**：每步产物落盘 `data/steps/{task_id}/{step}_result.json`（原子写 .tmp+replace）。`retry_task` 按 `_current_step` 从失败步骤重跑，前置产物复用——embed 失败只重跑 embed，不再重跑 parse/chunk。
- **前置缺失自愈**：某步发现前置产物缺失（异常状态）时，回退 `delay` 前置步骤而非报错。
- **去重时机后移**：`delete_source` 在「写库前（index 步）」执行，保证 index 重跑幂等，且 parse/chunk/embed 阶段旧数据仍可服务。
- **进度上报**：`progress` 在 EMBEDDING 阶段按 batch 精确更新；deepdoc 解析阶段也接入了真实进度回调（`build_document_tree(on_progress=...)` → `parse_into_bboxes(callback=...)`），前端「解析中」显示真实百分比。
- **失败重试**：可重试异常（网络/超时/429/5xx）指数退避自动重试；不可重试（402 余额/400 参数/文件损坏）直接 FAILED。
- **崩溃恢复**：`task_acks_late` + `visibility_timeout`（默认 600s），Worker 崩溃后未完成任务自动重投。
- **中断恢复**：关闭程序时 `__main__.py` 调 `interrupt_running()` 把「运行中」任务标记为「服务已关闭」；启动时 `server.py` 的 lifespan 再次调 `interrupt_running("服务重启，任务中断（可重试）")` 兜底清理上次遗留的运行态任务（强杀/崩溃也覆盖）。前端「最近入库任务」列表展示这些失败任务并提供重试入口，同时避免「同名文件正在上传」的误判（`has_running` 不再被卡住的任务占用）。
- **双队列**：`ingest.dual_queue=true` 时，`task_routes` 把 parse/chunk/index 路由到 `parse_queue`、embed 路由到 `embedding_queue`，独立扩容。
- **文件生命周期**：上传文件落盘到 `data/uploads/`，DONE 后清理，FAILED 保留供手动重试。

### 5.1 路径一：普通解析（父子块）

**流程**：`文件 → 提取纯文本 → 父子块切分 → 子块向量化 → 写入 Milvus`

**代码在哪**：`db_service.py` 的 `insert_documents`，底层 `embedding.py` 的 `embed_with_parent_child`

**核心逻辑**：

```python
# db_service.py（简化）
def insert_documents(text: str, source: str = "", on_progress=None) -> dict:
    emb = get_embedding()
    store = get_store()
    # 1. 父子块切分 + 子块向量化（一步完成）
    #    on_batch 回调 → 上报 EMBEDDING 阶段进度（供异步入库状态机）
    _emit(on_progress, "CHUNKING", 0)
    result = emb.embed_with_parent_child(
        text, source=source,
        on_batch=lambda done, total: _emit(on_progress, "EMBEDDING", int(done / total * 100)),
    )
    # 2. 写入 Milvus（父块存原文，子块存向量）
    _emit(on_progress, "INDEXING", 0)
    if hasattr(store, "insert_parent_child") and store._use_pc:
        store.insert_parent_child(source, result["parent_chunks"], result["child_chunks"])
    else:
        store.insert([c["text"] for c in result["child_chunks"]],
                     [c["vector"] for c in result["child_chunks"]])
```

**文件格式支持**：`.txt / .md / .py / .json`（纯文本直接读）+ `.pdf / .docx / .pptx / .epub / .odt / .eml`（chunklet 提取）+ `.xlsx`（openpyxl）+ `.rtf`（striprtf）。见 `embedding.py` 的 `_read_file_content`。

### 5.2 路径二：增强解析（文档树）

这是本系统的**核心亮点**。流程分五步，每步一个文件：

```
文件
  ↓ ① deepdoc 解析（deepdoc_extractor.py + deepdoc/）
扁平元素列表（标题/段落/表格 + 版面坐标）
  ↓ ② 结构归位（structure_resolver.py）
文档树（TreeNode 嵌套结构）
  ↓ ③ 摘要（summarizer.py）
带章节摘要 + 文档主旨的树
  ↓ ④ 切块（chunk_builder.py）
Retrieval Chunk 列表
  ↓ ⑤ 存储（db_service.py + tree_store.py + milvus_store.py）
SQLite 存树 + Milvus 存 chunk 向量
```

**统一入口**：`db_service.py` 的 `insert_documents_structured`（一体化）。Phase 3 步骤化任务复用同一套逻辑，拆成 `parse_step`/`chunk_step`/`embed_step`/`index_step` 四个纯函数（产物可 JSON 序列化落盘），供 Celery 步骤化编排，见 5.0。

```python
def insert_documents_structured(filepath, source="", on_progress=None):
    # 1. 结构归位
    _emit(on_progress, "PARSING", 0)
    root = structure_resolver.build_document_tree(filepath)
    # 2. 章节摘要 + 文档主旨摘要（LLM，失败不阻塞；封装为 _generate_summaries）
    abstract = _generate_summaries(root)
    # 3. 结构树 → chunk（纯内存操作，失败时尚未写任何库）
    _emit(on_progress, "CHUNKING", 0)
    chunks = chunk_builder.build_chunks(root)
    # 4. chunk 向量化（最昂贵、最易失败的步骤，置于所有写库操作之前；
    #    失败时无残留，天然保持 SQLite 与 Milvus 一致）
    _emit(on_progress, "EMBEDDING", 0)
    vectors = emb.embed_texts(
        [c["text"] for c in chunks],
        on_batch=lambda done, total: _emit(on_progress, "EMBEDDING", int(done / total * 100)),
    )
    # 5. 存树到 SQLite
    _emit(on_progress, "INDEXING", 0)
    doc_id = tree_store.save_tree(root, source=source, abstract=abstract)
    # 6. 写入 Milvus；失败回滚树库，避免「树库有文档但 Milvus 无 chunk」不一致
    try:
        store.insert_chunks(source, chunks)
    except Exception:
        tree_store.delete_document(doc_id)
        raise
```

下面逐个展开这五步。

---

### 5.2.1 第一步：deepdoc 解析（把 PDF 拆成"元素"）

**是什么**：用 RAGFlow 的 deepdoc 引擎，把 PDF/DOCX 解析成**扁平的元素列表**——每个元素带 `layout_type`（标题/段落/表格/图）、文本内容、版面坐标（页码 + 位置）。

**为什么用 deepdoc 而不是普通提取**：普通提取只拿到"文字流"，丢失了**表格结构、标题层级、版面位置**。deepdoc 能识别这些结构。

**代码在哪**：`deepdoc_extractor.py`（分发）、`deepdoc/`（引擎）

**核心逻辑**：

```python
# deepdoc_extractor.py（示意）
def extract(file_path, ext):
    parser = PARSERS.get(ext)          # .docx→DocxParser / .pdf→RAGFlowPdfParser
    elements = parser.parse(file_path)  # 扁平元素列表
    return clean_elements(elements)     # 剔除版面坐标标签
```

---

### 5.2.2 第二步：结构归位（元素 → 树）

**是什么**：把上一步的**扁平元素列表**，按标题层级还原成**文档树**。

**标题层级来源**（不同格式不同来源）：

| 格式 | 标题怎么识别 |
|---|---|
| .pdf | 书签大纲（outline）优先，标题元素 + 编号/字号兜底 |
| .docx | Word 样式（Heading 1/2/3） |
| .md | `#` / `##` / `###` |
| .html | `h1`~`h6` |
| .txt | 无层级，退化为扁平 |

**代码在哪**：`structure_resolver.py` 的 `StructureResolver.resolve`

**核心逻辑**（建树算法）：

```python
def _build_tree(self, headings, content, doc_id, doc_title):
    root = self._new_node("document", ...)
    # 1. 用「栈」按层级建 section 节点
    stack = [root]
    for h in headings:                      # 标题已按顺序排序
        node = self._new_node("section", title=h["title"], level=h["level"])
        while len(stack) > 1 and stack[-1].level >= node.level:
            stack.pop()                     # 同级或更高级标题 → 出栈
        stack[-1].children.append(node)     # 挂到最近的父章节
        node.section_path = stack[-1].section_path + [node.order]
        stack.append(node)
    # 2. 内容归位：每个正文块归到「顺序在它之前的最近标题」
    for c in content:
        target = 最近的 section
        target.children.append(段落/表格节点)
    return root
```

**关键：结构残缺时 LLM 重建目录树**。如果文档没有标题（如纯 TXT、无大纲的 PDF），系统会判断"结构残缺"，然后用 LLM 从正文里**重建目录树**（分批识别哪些块是章节标题）。失败则静默回退扁平树，不阻塞入库。见 `_should_reconstruct` / `_reconstruct_headings`。

---

### 5.2.3 第三步：章节摘要 + 文档主旨

**是什么**：用 LLM 给每个章节生成一句话摘要，给整篇文档生成"主旨 + 关键词"。

**为什么**：
- **章节摘要**：检索时的"廉价铺垫"——不用读全文就能快速判断章节相关性。
- **文档主旨**：跨文档路由的依据——问题来了先判断"该去查哪篇文档"。

**设计约束**：摘要只用于**路由/评估**，最终答案仍用**原文**（绝不用摘要替代原文，否则信息损失 + 幻觉）。

**代码在哪**：`summarizer.py`

```python
# summarizer.py（简化）
def summarize_tree(root, llm, max_workers=4):   # 章节摘要（并发）
    # 章节摘要之间完全独立（每个 section 只读自己的直属叶子），且 LLM 调用是网络
    # IO，故用线程池并发：串行 N 个 section = N×单次耗时，并发 = N/max_workers
    # ×单次耗时。结果与串行完全一致（纯无损，只是调度顺序不同）。
    sections = 收集所有 section
    with ThreadPoolExecutor(max_workers) as ex:
        for section in sections:
            leaves = 该 section 直属叶子文本
            section.summary = summarize_section(section.title, leaves, llm)  # 一句话摘要

def summarize_document(root, llm):      # 文档主旨
    # 基于「文档标题 + 章节目录」生成 JSON：
    # {"abstract": "一句话主旨", "keywords": ["k1", "k2", "k3"]}
    return json_str   # 存进 documents.abstract
```

> **注意**：摘要只在**入库时**生成一次，检索阶段绝不实时调 LLM（避免检索变慢、成本爆炸）。
>
> **并发度**由 `config.json` 的 `summary.concurrency` 控制（默认 4），设 1 退化为串行（原行为）。章节摘要并发是增强解析提速的主要手段（实测 12 节提速 4 倍）。
>
> **PDF 解析的 GPU 加速**：deepdoc 的 OCR/版面/表格识别模型走 `onnxruntime-gpu` 的 CUDA EP，大 PDF 解析实测提速约 2.2 倍（50 页论文 330s → 147s）。解析分辨率由 `config.json` 的 `deepdoc.zoomin` 控制（默认 3，最高精度；降为 1/2 可再提速约 20%，但可能漏检小字号内容）。

---

### 5.2.4 第四步：树切块（Retrieval Chunk）

**是什么**：把文档树的**叶子节点**（段落/表格/图）切成检索单元（chunk）。

**切分规则**（`chunk_builder.py`）：

| 规则 | 行为 |
|---|---|
| 叶子 ≤ `max_chars`（800） | 独立成 1 个 chunk |
| 叶子 > `max_chars` | 按句子边界切成多个 chunk（不切断句子） |
| 同 section 连续短叶子（≤ `min_chars`=150） | 合并成 1 个 chunk |

**关键字段**（chunk 与树的关系）：

```python
{
    "chunk_id": "doc_xxx:chunk:12",      # 稳定 ID
    "text": "...",                        # 正文
    "parent_node_id": "doc_xxx:n0001",    # 挂回树节点（上下文恢复锚点）
    "source_node_ids": [...],             # 完整出处（合并场景为多个叶子）
    "doc_id": "doc_xxx",                  # 所属文档
    "section_path": [1, 2],               # 章节路径（如第1章第2节）
    "chunk_seq": 12,                      # 全局阅读顺序（邻近块扩展用）
}
```

**代码在哪**：`chunk_builder.py` 的 `build_chunks`

---

### 5.2.5 第五步：向量化 + 双存储

**是什么**：把 chunk 文本向量化，然后**分两处存**：
1. **SQLite**（`tree_store.py`）：存树结构 + 节点原文
2. **Milvus**（`milvus_store.py`）：存 chunk 的向量 + chunk 文本

**为什么分离**：SQLite 擅长"结构查询"（顺着树找孩子/父亲/章节路径），Milvus 擅长"向量相似检索"（语义召回）。各用所长。

---

## 6. 存储层：SQLite 树库 + Milvus 向量库

### 6.1 SQLite 树库（tree_store.py）

**职责**：存文档树的结构 + 每个节点的原文。

**核心查询原语**（纯树导航检索的基础）：

| 函数 | 作用 |
|---|---|
| `save_tree` / `load_tree` | 存/取整棵树 |
| `get_document_root_id` | 找文档根节点 |
| `get_children` | 取某节点的直接子节点 |
| `get_node` | 取某节点的详情（title/text/summary） |
| `match_sections` | 关键词匹配章节标题 |
| `get_top_ancestor` | 取某节点的顶层祖先 |
| `get_section_path_titles` | 把数字路径（如 [1,2]）还原成章节标题 |
| `get_subtree_stats_all` | 一次性算全文档所有 section 的子树体量 |
| `get_representative_texts_all` | 一次性算所有 section 的代表性叶子文本 |

### 6.2 Milvus 向量库（milvus_store.py）

**职责**：存 chunk 向量，提供向量检索 / 混合检索。

**父子双 Collection**（普通解析路径）：
- `parents`：父块原文（不向量化）
- `children`：子块向量 + 文本（向量化）

**结构树 chunk**（增强解析路径）：chunk 带 `doc_id`，检索时直接返回 chunk 文本，章节路径由上层 `tree_store` 恢复。

---

## 7. 检索：把答案「找」出来

系统支持**三种检索模式**，前端「设置 → 检索模式」三选一：

| 模式 | 原理 | 适用 |
|---|---|---|
| **vector** | 纯向量语义召回 | 通用 |
| **hybrid** | 向量 + BM25 融合 | 需要精确关键词 |
| **tree** | 纯树导航（不碰向量） | 结构化文档 |

### 7.1 混合检索（hybrid）

**是什么**：dense 向量召回 + BM25 稀疏召回，RRF 融合排序。

**代码在哪**：`milvus_store.py` 的 `search_hybrid`

**核心逻辑**：

```python
def search_hybrid(self, query_vector, query_text, top_k=5):
    limit = top_k * 2
    # 两路召回
    req_dense  = AnnSearchRequest([query_vector], "vector",
                                  {"metric_type": "COSINE"}, limit)   # 语义
    req_sparse = AnnSearchRequest([query_text], "sparse_vector",
                                  {"metric_type": "BM25"}, limit)      # 关键词
    # RRF 倒数融合排序
    results = self.client.hybrid_search(
        collection_name=self.child_col_name,
        reqs=[req_dense, req_sparse],
        ranker=RRFRanker(k=60), limit=limit)
    # RRF 分数归一化到 [0,1]（否则量级与 COSINE 不可比，会误判相关性）
    return self._format_child_hits(results[0], top_k)
```

**上层入口**：`db_service.py` 的 `search_documents(hybrid=True)`，检索后还会做两件事：
1. **章节路径恢复**：沿树把数字路径还原成章节标题（`get_section_path_titles`）
2. **邻近块扩展**：命中 chunk 后，补充同 section 相邻的 chunk（解决答案跨块问题）

### 7.2 纯树导航检索（tree）

**是什么**：把"检索"重新定义为"**在文档树上的逐步探索**"，**完全不碰向量召回**。用「reranker 打分 + LLM 决策」逐层 descend（深入）/ backtrack（回溯）。

**类比**：你拿着一本书的目录，一层层翻找——先看有哪些章节，判断哪个章节相关，钻进去再看小节，最后找到相关段落抄下来。

**代码在哪**：`tree_retrieval.py` 的 `tree_navigate`（唯一实现，输出统一 dict）

**八条设计原则**（保证状态机正确性的根基）：

1. 单一 stack（栈顶 = current），不搞双调度
2. NodeState 有 `exhausted`（子节点全处理完）
3. `is_searchable` = status ∈ {unvisited, expanded}
4. 叶子判断写死成节点类型（section=容器，其余=叶子）
5. 叶子自动 Leaf Rerank → read，**不让 LLM 决定 read**
6. Stop Policy 由**代码**触发，不让 LLM 决定何时停
7. LLM 只提议动作（descend/backtrack），代码执行和裁决
8. trajectory 保留完整搜索轨迹

**核心状态机**：

```python
def tree_navigate(query, reranker, llm, doc_id=None):
    state = TreeNavState()
    entries = _init_entries(query, doc_id=doc_id, llm=llm)   # ① 定位入口
    for e in reversed(entries):
        state.stack.append(e)                                # 逆序压栈

    while state.stack:
        ns = state.nodes[state.stack[-1]]                    # 栈顶 = current
        if is_leaf_kind(ns.kind):                            # 叶子：自动打分→读/丢
            _leaf_rerank_read(...); state.stack.pop(); continue
        if ns.status == "unvisited":                         # 首次：展开子节点
            _expand(state, ns); ns.status = "expanded"
        section_children, leaf_children = _split_children(state, ns)
        _leaf_rerank_batch(state, query, reranker, leaf_children, ...)   # ② 叶子批量打分
        _node_rerank(state, query, reranker, section_children, ...)      # ③ 章节剪枝
        searchable = [c for c in section_children if _is_searchable(...)]
        if not searchable:                                   # 无路可走→回溯
            ns.status = "exhausted"; state.stack.pop(); continue
        if _check_stop(state, ...): break                    # ④ Stop Policy（代码触发）
        target = _choose_next(state, query, llm, ns, searchable, ...)    # ⑤ LLM 决策
        if target is None:                                   # backtrack
            ns.status = "exhausted"; state.stack.pop(); continue
        state.stack.append(target)                           # descend（深入）
    return {"evidences": sorted(evidences), ...}
```

**双层三区间剪枝**（核心决策点）：

| 阶段 | 评分对象 | 三区间裁决 |
|---|---|---|
| Node Rerank | section 的 title + summary + 代表性叶子文本 | `< min` 剪枝；`[min, high)` 模糊区标记交 LLM；`≥ high` 保留 |
| Leaf Rerank | leaf 的 text | `< min` 剪枝；`[min, high)` read 但标记模糊；`≥ high` read |

**Stop Policy**（由代码触发，LLM 不决定）：只有「非模糊 + 非相邻补读 + 分数 ≥ high」的叶子才算高分证据，凑够 `min_evidences`（默认 2）条就停。

**相邻叶子补读**：命中叶子后，按 `neighbor_window`（默认 ±1）读同父相邻叶子，补全跨段落/表格的答案。

### 7.3 文档级路由（多文档场景）

**是什么**：知识库有多篇文档时，先判断"该去哪篇文档"，再在选中的文档里做树导航。

**流程**：
```
列出所有文档卡片（主旨+关键词，入库时生成）
  → LLM 选出 top-1 文档
  → 在该文档里 tree_navigate(doc_id=top1)
  → 证据不足（< 3 条）→ 自动 fallback top-2/top-3 合并
```

**代码在哪**：`tree_retrieval.py` 的 `route_to_docs` / `retrieve_by_doc_routing`

### 7.4 文档信息增量检索（DocNovelty）

**是什么**：解决文档路由的「早停漏洞」——旧逻辑是「第一本找到 ≥ 3 条证据就停，不再翻第二本」，但**条数够 ≠ 信息覆盖完整**。当答案需要跨文档拼合（A 文档讲定义、B 文档讲应用、C 文档讲局限），第一本找够 3 条就停，B/C 独有的知识就漏了。

DocNovelty 把停止判据从「数量」升级为「信息增量」：**每篇候选文档都判断「它的核心知识是否已被当前证据覆盖」——覆盖了就跳过，没覆盖就继续搜。**

**为什么**：分清两个概念——`retrieval relevance`（文档与问题相关）和 `information novelty`（文档带来新信息）。候选文档都是「与问题相关」的（已按相关度排序），但只有「带来新信息」的才值得继续搜。

**怎么做**（核心判断方向）：

```
判断方向必须是 D → E（候选文档主旨 → 已有证据）
   ✗ 错误：question → D 和 question → E（两者都被 question 筛出，天然都高相关，无法区分）
   ✓ 正确：query = D 的主旨+关键词，documents = 已有证据集合
```

**核心逻辑**（`tree_retrieval.py`）：

```python
def judge_doc_novelty(card, coverage_evidence, reranker, coverage_high=0.55):
    """DocNovelty：候选文档相对当前证据的信息重复度判断（V1 只二分）。"""
    if reranker is None or not coverage_evidence:
        return "uncovered"                          # 边界：一律保守搜
    query = (card.get("abstract") or "").strip() or (card.get("title") or "").strip()
    if not query:
        return "uncovered"                          # 卡片空 → 保守搜
    docs = [f"{c['path']} {c['summary']} {c['text']}".strip()
            for c in coverage_evidence if ...]      # 证据文本拼接
    res = reranker.rerank(query, docs)              # 批量打分
    max_score = max((float(r["score"]) for r in res), default=0.0)
    return "covered" if max_score >= coverage_high else "uncovered"   # 二分
```

**三个关键设计**（方案 C V1 定稿）：

1. **宁可多搜、绝不误跳**：只有 reranker 明确给出「高度覆盖」（≥ `coverage_high`）才判 covered，其余一律 uncovered。因为「多搜一篇 = 性能问题」，而「少搜一篇 = 答案缺失」，系统首选不漏信息。

2. **Coverage Evidence 与最终 Evidence 分离**：判断基准用独立的轻量「知识覆盖集合」（`build_coverage_evidence` 产出，排除相邻补读 + 按 score 取 top_k + 只留 path/summary/text[:150] 三字段），不受最终 top_k 截断影响——避免「D1 的后几条知识不在 top_k 里，导致 D2 被误判 uncovered 而重复搜索」。

3. **`coverage_max_skip` 只是启发式早停**：文档排序是「相关度排序」而非「信息增量排序」，连续 N 篇 covered 提前停**只是成本控制，不保证后面无新信息**。最终停止权仍归 Evaluator。

**主循环逻辑**（`retrieve_by_doc_routing` 覆盖驱动遍历）：

```python
evs = []                  # 最终 Evidence
coverage_evidence = []    # 知识覆盖集合（独立维护）
skip_streak = 0

for i, did in enumerate(doc_ids):
    if i == 0:                                          # 第一篇永远搜（无对比基准）
        evs += tree_navigate(doc_id=did)
        coverage_evidence = build_coverage_evidence(evs, top_k)
        continue

    verdict = judge_doc_novelty(_find_card(did), coverage_evidence, reranker, coverage_high)

    if verdict == "covered":
        skip_streak += 1
        if skip_streak >= coverage_max_skip:            # 启发式早停
            break
    else:
        skip_streak = 0
        evs += tree_navigate(doc_id=did)                # 有新增信息，继续搜
        coverage_evidence = build_coverage_evidence(evs, top_k)
```

**代码在哪**：`tree_retrieval.py` 的 `build_coverage_evidence` / `judge_doc_novelty` / `retrieve_by_doc_routing`（改造后）

**配置**（`config.json` → `agentic.doc_router`）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `novelty_enabled` | true | 方案 C 总开关（关闭则回退旧的条数早停逻辑） |
| `coverage_high` | 0.55 | ≥ 判 covered（宁可偏高，宁可多搜） |
| `coverage_max_skip` | 2 | 连续 covered 提前停阈值（启发式，非正确性保证） |
| `coverage_evidence_top_k` | 5 | 参与覆盖判断的高分证据条数 |

### 7.5 三级降级（tree_search）

树检索结果不足 `top_k` 时，逐层补齐：

```
① 文档级路由 + 单文档树检索（不碰向量）
   ↓ 不足
② 章节定位检索（match_sections 命中章节 → section_path 过滤，仍不碰向量）
   ↓ 不足
③ 以文检文协同（树命中正文 + 原 query 增强 → hybrid 补齐）
```

**代码在哪**：`tree_retrieval.py` 的 `tree_search`

---

## 8. Agentic RAG（多步检索循环）

### 8.1 是什么

把复杂问题拆成多个"信息需求"，显式维护「需求 → 证据 → 缺口」状态，缺口驱动循环检索，直到证据充分或触发停止。

**与 rag 模式的区别**：rag 是"固定一步检索"，agentic 是"多步循环，直到答好为止"。

### 8.2 六大组件

**代码在哪**：`agentic_rag/` 子包，`agent.py` 编排

| 模块 | 职责 |
|---|---|
| `router.py` | 复杂度判断（SIMPLE / MEDIUM / COMPLEX） |
| `planner.py` | 拆解为 Requirements（对比型拆单点 + deferred 展开 + 关系型 synthetic 归纳） |
| `state.py` | 显式状态 AgentState |
| `controller.py` | Gap → Action 决策 |
| `executor.py` | 执行 Action（SEARCH / READ / REFINE / WEB_SEARCH） |
| `evaluator.py` | 证据评估（SUPPORTED/PARTIAL/MISSING） |
| `stopping.py` | 三层停止策略 |
| `synthesizer.py` | 带引用合成（无证据时生成诚实的"无法回答"） |
| `retriever.py` | 检索抽象 + 模式路由 |

### 8.3 核心闭环

```python
# agent.py（示意）
def run_agentic(question, reranker, llm, ...):
    complexity = classify_complexity(question)   # LLM 主导判断
    if complexity in (SIMPLE, MEDIUM):
        return _run_light(question, ...)         # 轻量循环
    requirements = plan(question)                # 拆解需求
    state = AgentState(question, requirements)
    while not stopping_rule(state):              # 三层停止规则
        action = controller.decide(state)        # 决策
        executor.execute(state, action, ...)     # 执行
        evaluator.evaluate(state, ...)           # 评估
    return synthesize(state, llm)                # 带引用合成
```

### 8.4 关键设计

**1. 复杂度判断交给 LLM**：规则只做明确 COMPLEX 的快速短路，其余交给 LLM 判断"是否需要拆解"（能识别规则覆盖不到的语义复杂度）。

**2. deferred 展开（区别类问题）**：像"角色型 agent 有什么人格？区别是什么？"这类问题，先检索拿到**真实清单**（三种人格），再生成**逐项对比**的 query 逐个检索。这避免用笼统的"区别"去硬检索。代码在 `planner.py` 的 `_mark_deferred_requirements` / `expand_deferred`。

**3. synthetic 归纳（关系/联系类问题）**：像"角色型 agent 与 RAG 的关系"这类需求，知识库通常**没有直接论述二者关系的独立段落**（RAG 与 agent 是两套各自独立的文档），若把它当独立检索目标，Controller 会反复 SEARCH/READ/WEB_SEARCH 空转。解法：planner 识别「关系/联系/关联」信号（LLM 规则 + 正则兜底），把该需求标记为 `synthetic`，依赖 A、B 的单点需求；它**不进入 gap 循环**，待 A、B 都 SUPPORTED 后由 Synthesis 归纳二者关系。代码在 `planner.py` 的 `_mark_synthetic_requirements` + `state.py` 的 `resolve_synthetic`。识别要求「关系信号 + 实体连接词」同时出现，避免误伤「关联规则」「数据融合」这类单实体技术名词。

**3. 三层停止策略**（纯规则，无 LLM Judge）：

| 层 | 条件 |
|---|---|
| Hard Stop | 轮数 / 工具调用次数超上限 |
| Sufficiency | 所有高重要度需求均 SUPPORTED |
| No-progress | 连续 N 轮无需求状态升级 |

**4. 模型分层**：Synthesis 用强模型 `llm`（保留思考）；Planner/Evaluator/Controller 用 `tool_llm`（关思考，快）。

**5. 无证据时不编造**：`synthesizer.py` 在无验证证据时，用 LLM 生成一个诚实、有针对性的"无法回答"回复（含换问法建议），而不是返回固定的 `[未获取到回答]`。

**6. Controller 硬约束：0 证据禁止 ANSWER**（`controller.py`）：对「单点事实题」，Controller LLM 常自认已知答案、首轮就误判「信息已充分」直接合成，导致 0 证据 → citations 为空。故在 `choose_action` 加硬约束——**0 证据但仍有缺口时，强制先 SEARCH 一次**（LLM 决策在其后），彻底杜绝「没检索就回答」。若检索真的返回空，由 stopping 的 no-progress 机制兜底终止，不会死循环。

---

## 9. Reranker 重排序

**代码在哪**：`reranker.py`

**流程**：`Embedding 粗召回 Top 20 → Reranker 精排 Top 5 → LLM`

**两种后端**：
- 本地：`BAAI/bge-reranker-v2-m3`（HF Transformers 交叉编码器）
- 远程：HTTP API

**核心逻辑**：

```python
device = "cuda" if torch.cuda.is_available() else "cpu"   # 自动走 GPU
# 本地模型输出未归一化 logits，sigmoid 映射到 [0,1]
score = 1.0 / (1.0 + math.exp(-logit))
```

| 分数 | 含义 |
|---|---|
| `> 0.9` | 高度相关 |
| `≈ 0.5` | 模棱两可 |
| `< 0.1` | 不相关 |

> **预热机制**：`server.py` 的 lifespan 里 `_warmup_reranker_async` 会在后台线程预热加载 reranker 模型，避免首次对话卡在模型加载上。

---

## 10. LLM 封装与模型工厂

### 10.1 LLM 封装（llm.py）

**是什么**：统一封装两种模型后端。

| 后端 | 协议 | 说明 |
|---|---|---|
| ChatOpenAI | OpenAI 兼容 | 标准流式（`.stream()`） |
| DoubaoLLM | 火山方舟 Responses API | 手动 SSE 解析（豆包专用） |

**核心函数 `create_chat_model`**：按 `protocol` 参数选择后端；支持 `disable_thinking`（给 reasoning 模型关思考，决策类场景要快）。

### 10.2 模型工厂（llm_factory.py）

**是什么**：统一入口 `get_model(*kinds, answer)`，带进程级缓存。

**解决什么问题**：原来 8 处模块各自复制"按回退链取模型"的逻辑，且每次新建实例浪费连接。现在统一成一个入口。

```python
decision = get_model("tool_llm", "llm")     # 决策类（回退链：tool_llm → llm）
summary  = get_model("summary", "tool_llm", "llm")
answer   = get_model("llm", answer=True)    # 生成答案（温度更高，不关思考）
```

### 10.3 DSML 解析器（dsml_read.py）

**是什么**：DeepSeek V4 等 agentic 模型不遵循标准 function calling，而是把工具调用输出成 DSML 文本标记。本模块用正则解析这些标记，转成 OpenAI 兼容的 tool_calls。

**代码在哪**：`dsml_read.py` 的 `parse_dsml_tool_calls`

---

## 11. MCP 集成与工具管理

### 11.1 核心架构

**所有工具都注册在 MCP 服务器上，主程序不持有工具定义，调用工具一律通过 `mcp_service/tool_bridge.py` 连接 MCP 执行。**

```
主程序（rag 工具决策 / agentic 联网搜索）
        │ 唯一入口
        ▼
mcp_service/tool_bridge.py ──连接──▶ MCP 服务器（6 个工具）
```

### 11.2 六个 MCP 工具

| 工具 | 功能 |
|---|---|
| `web_search(query, num)` | 联网搜索（Google News RSS → Bing → 百度） |
| `calculate_pi(digits)` | 计算 π（Chudnovsky 算法） |
| `calculate_expression(latex, precision)` | 计算数学算式（支持 LaTeX） |
| `search_knowledge_base(query, top_k, mode)` | 知识库检索（vector/hybrid/tree） |
| `list_knowledge_documents()` | 列出已入库文档 |
| `get_knowledge_toc()` | 获取知识库目录结构 |

### 11.3 工具桥接（tool_bridge.py）

**是什么**：把 MCP 工具动态转换成 langchain 工具，供 `bind_tools` 决策。

```python
def get_mcp_tools_as_langchain(server_name=None, enabled_only=True):
    tools = []
    for t in manager.list_tools(server_name)["tools"]:
        if enabled_only and not _is_tool_enabled(t["name"]):   # 启用过滤
            continue
        ArgsModel = _schema_to_model(t["name"], t["input_schema"])  # schema→pydantic
        tools.append(StructuredTool.from_function(
            func=lambda **kw, n=t["name"]: call_tool_by_name(n, kw),  # 执行走 MCP
            name=t["name"], description=t["description"], args_schema=ArgsModel))
    return tools
```

### 11.4 工具调用决策（模型自主决策）

**是什么**：rag 链路在检索评估之后，由一个「工具决策模型」（`tool_llm`）判断「回答这个问题是否需要调用外部工具（联网搜索 / 数学计算等）」，需要就调用，不需要就不调用。

**为什么是自主决策而非写死规则**：早期版本曾在提示词里写死「纯数学问题调 calculate_expression」「query 用 2~6 个词」等规则，结果是规则覆盖不全、误判频繁，且与模型判断打架。现改为**把可用工具列表 + 知识库检索情况如实告知决策模型，由模型自主判断**——需要什么信息就调什么工具，不需要就不调，不预设任何"什么问题调什么工具"的规则。

**怎么做**（`rag_graph.py` 的 `_decide_and_run_tools`）：
- 决策模型收到：可用工具清单（含每个工具的参数 schema）+ 知识库检索结果摘要 + 用户问题 + 对话历史
- 模型自主决定调哪些工具（或都不调），输出 function call
- 执行工具后，把结果**独立成段**拼进生成模型的 system prompt

**关键设计——工具结果独立成段**：工具调用/联网结果放在 `=== 工具调用结果 ===` 独立区块，并明确标注「这是可信结果，可直接用于回答，无需标注来源编号」。早期版本把工具结果混在 `=== 检索内容 ===` 里，被「只依据检索内容 + 没有依据不写 + 标注来源编号」的约束压制，导致明明算了结果却仍答「无法回答」。独立成段后，生成模型能正确使用工具结果。

**代码在哪**：`rag_graph.py` 的 `_decide_and_run_tools`（决策）、`generate` 节点里 `tool_section` 拼接（独立成段）。

### 11.5 MCP 管理页面（路由 /#/mcp）

- **服务器列表**：启动 / 停止 / 删除 / 状态（含 pid）
- **工具**：每个工具**独立**启用/禁用 + 调试（按 input_schema 动态生成参数表单）
- **服务器日志**：实时查看（自动刷新，保持滚动位置）
- **工具决策模型**：下拉切换 + 内联新增（`tool_llm`）

> 旧入口 `/mcp_page`、`/tools_page` 已保留，但都是 302 重定向到 `/#/mcp`（Vue SPA 路由）。

> 工具决策模型的选择已从主程序设置页移除，**统一在 MCP 管理页配置**。

### 11.6 `--mcp` / `--celery` 参数（同步启停）

```bash
python __main__.py --mcp             # 主程序启动时同步启动 MCP，退出时同步关闭
python __main__.py --celery          # 同步启动 Celery Worker（异步入库），退出时同步关闭
python __main__.py --mcp --celery    # 三个服务一起启停
```

`__main__.py` 里 `_start_mcp` / `_stop_mcp` 调用 `manager.start_server/stop_server`，`_start_celery` / `_stop_celery` 拉起/关闭 Worker，并注册到信号 handler，保证退出时子进程一并终止。

**Ctrl+C 退出会杀整棵进程树**：Windows 上 `Popen.terminate()` 只杀直接子进程本身，而 celery worker / MCP 服务器都会再 spawn 子进程（billiard 进程池等），之前会导致孤儿进程残留。现改为用 `taskkill /T /F` 杀整棵进程树（Unix 用 terminate→kill），保证主程序、Worker 及其子进程、MCP 及其子进程全部干净退出，无残留 python 进程。

---

## 12. 前端页面

前端已从单文件 HTML 重构为 **Vue3 + Naive UI + Pinia + Vite** 工程（源码 `frontend/`，产物 `static/dist/`），是一个单页应用（SPA）：主界面与 MCP 管理页是同一个应用里的两个路由。

### 12.1 技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| 框架 | Vue 3（`<script setup>`） | 组合式 API |
| UI 组件库 | Naive UI | 主题定制强，对齐暗色/亮色 |
| 状态管理 | Pinia | 替代早期单文件的全局变量 + 手动 localStorage 同步 |
| 路由 | Vue Router（hash 模式） | `/#/` 主界面、`/#/mcp` MCP 管理 |
| 图标 | `@lucide/vue` | SVG 图标 |
| Markdown/公式 | `markdown-it` + `katex` | 见 12.4 |
| 构建 | Vite | 产物输出到 `static/dist/` |

**主题系统**：亮 / 暗 / 跟随系统 三态切换，状态存 `localStorage['rag-theme']`。CSS 变量在 `src/styles/global.css` 定义（亮暗两套），Naive UI 组件主题在 `src/theme/` 覆盖。SPA 内主界面与 MCP 页天然共享同一主题状态，无跨页同步问题。

### 12.2 主界面（ChatView.vue）

- **对话区**：三模式切换（知识库问答 / Agentic 检索 / 直接对话）+ 流式回答 + 思考过程折叠 + Agent 工作台 + 引用来源卡片
- **对话历史侧栏**：新建 / 切换 / 删除对话（存 SQLite，见 `chat_history.py`）
- **问题追溯条**：右侧竖条标记每条用户提问，点击可跳转定位
- **设置弹窗**：管理 LLM / Embedding / Reranker / Summary / tool_llm 模型 + 检索模式 + 数据库 + reranker 加载
- **右侧面板**：数据库管理（文件树/子块浏览与删除）+ 文件上传 + 系统提示词 + 服务日志
- **顶栏**：主题切换 + 「MCP 管理」入口 + 数据库连接状态

**上传进度条**：每个上传文件下方有进度条，按入库阶段分段展示——排队/解析/切分/写索引用流动动画（shimmer），向量化（EMBEDDING）显示精确百分比（映射到 42%~88%），完成变绿、失败变红。

**最近入库任务列表**：页面加载时拉取 `/upload/tasks` 展示最近任务，失败任务（含「上传途中关程序」导致的中断）显示红色错误 + 「重试」按钮，点击调 `/upload/{task_id}/retry` 从失败步骤继续入库。

### 12.3 MCP 管理页（McpView.vue，路由 /#/mcp）

见上节 11.4。后端保留 `/mcp_page`、`/tools_page` 入口，现在都是 302 重定向到 `/#/mcp`。

### 12.4 流式渲染与 LaTeX

**流式渲染优化**（避免长回答卡顿）：
- 流式期间用**轻量纯文本渲染**（只转义 + 换行，不跑完整 Markdown），`done` 时一次性完整格式化
- token 用 `requestAnimationFrame` 节流合并（同一帧内多个 token 只触发一次渲染）
- 消息按唯一 id 做 key，切换/新建对话不会复用旧组件状态
- 思考过程、Agent 工作台、正文、引用来源分区独立更新

**LaTeX 公式渲染**（`utils/markdown.js` + `katex`）：支持**四种分隔符**，块级/行内全覆盖：

| 类型 | 分隔符 |
|---|---|
| 块级 | `\[...\]` 与 `$$...$$` |
| 行内 | `\(...\)` 与 `$...$` |

渲染流程：先按「块级 → 行内」顺序把公式替换为占位符（`$$` 必须先于 `$` 处理，避免被拆成两个 `$`），跑完 Markdown 再还原为 KaTeX HTML。占位符用 `\u0001`（SOH 控制字符）——`\u0000` 会被 markdown-it 替换成 `�` 导致失效。单个 `$`（如货币）不会误判，只有成对 `$...$` 才识别为公式。

**后端配合**：三个对话模式（rag/direct/agentic）的提示词里都要求模型用 `\(...\)`（行内）和 `\[...\]`（块级）输出公式，前端四种分隔符兜底，双重保障。

---

## 13. 配置体系

配置分三个文件，均在 `config/`：

| 文件 | 用途 |
|---|---|
| `config.json` | 系统配置（检索/agentic/工具/联网/结构重建） |
| `models.json` | 模型配置（llm/tool_llm/summary/embedding/reranker + current） |
| `db.json` | 数据库配置（Milvus + current） |

### 13.1 config.json 关键板块

**search（检索）**：`retrieval_mode`（vector/hybrid/tree）

**agentic（Agentic RAG）**：`max_iterations`、`no_progress_threshold`、`tree_nav.min_evidences`、`web_search.enabled` 等

**tools（MCP 工具启用/禁用）**：

```json
"tools": {
  "web_search": {"enabled": true},
  "calculate_pi": {"enabled": true},
  "calculate_expression": {"enabled": true}
}
```

**doc_router（文档路由）**：`fallback_top_n`、`min_evidences_fallback`

**ingest（异步入库）**：队列/重试/超时参数，代码不硬编码：

| 键 | 默认值 | 说明 |
|----|--------|------|
| `dual_queue` | `false` | 双队列开关：`true` 时 parse/chunk/index 与 embed 分流到两个队列 |
| `parse_queue` | `"parse_queue"` | 解析/切分/索引队列名（CPU 密集） |
| `embedding_queue` | `"embedding_queue"` | 向量化队列名（网络 IO 密集） |
| `max_retries` | `3` | 每步骤任务最大重试次数 |
| `retry_backoff_base` | `5` | 指数退避基数（秒）：base / base*2 / base*4 |
| `visibility_timeout` | `600` | worker 崩溃后未 ACK 任务重新投递等待时间（秒） |
| `worker_pool` | `"threads"` | `--celery` 启动 Worker 的池类型：`threads`（并发，默认）或 `solo`（串行，适合多 PDF 深度解析） |
| `worker_concurrency` | `4` | Worker 并发数（`worker_pool=threads` 时生效） |
| `redis_broker` | `"redis://127.0.0.1:6379/0"` | Celery broker 地址 |
| `redis_backend` | `"redis://127.0.0.1:6379/1"` | Celery result backend 地址 |

**summary（章节摘要）**：`enabled`、`concurrency`（并发度，默认 4）

**deepdoc（PDF 增强解析）**：`zoomin`（解析分辨率，默认 3 最高精度；降为 1/2 提速约 20%，可能漏检小字号）

### 13.2 models.json 结构

```json
{
  "llm": [{"name": "deepseek", "model": "deepseek-chat", "base_url": "...", "protocol": "openai"}],
  "tool_llm": [{"name": "tool-planner", "model": "deepseek-ai/DeepSeek-V4-Pro", "disable_thinking": true}],
  "embedding": [{"name": "bge-m3", "model": "BAAI/bge-m3"}],
  "reranker": [{"name": "bge-reranker-local", "type": "local", "model_path": "BAAI/bge-reranker-v2-m3"}],
  "current": {"llm": "deepseek", "tool_llm": "tool-planner", "reranker": "bge-reranker-local"}
}
```

> `tool_llm` 配 reasoning 模型（DeepSeek V4 Pro）必须 `disable_thinking: true`，否则决策/评估会先跑一大段思考，极慢。

---

## 14. API 接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/` | GET | Web 管理界面 |
| `/upload` | POST | 上传文件入库（**异步**，立即返回 `task_id`；`enhance=true` 走 deepdoc） |
| `/upload/tasks` | GET | 入库任务列表 |
| `/upload/{task_id}/status` | GET | 查询单个入库任务状态（status/progress/error/stats） |
| `/upload/{task_id}/retry` | POST | 重试失败的入库任务（步骤级：从失败步骤重跑，前置产物复用） |
| `/chat` | POST | 对话（SSE，`mode`：direct/rag/agentic，`retrieval_mode`：vector/hybrid/tree） |
| `/config` / `/config/select` | GET/POST | 模型/数据库配置 + 切换 |
| `/models` / `/dbs` | GET/POST/PUT/DELETE | 模型/数据库管理（PUT 编辑模型，API Key 留空表示不修改） |
| `/local/sources/rename` | PUT | 重命名文件（只改文件名，不改内容） |
| `/conversations` | GET/POST | 对话历史管理 |
| `/reranker/status` / `/reranker/load` | GET/POST | 本地 reranker 状态/加载 |
| `/logs` | GET | 服务日志 |
| `/mcp_page` | GET | MCP 管理页面（302 重定向到 `/#/mcp`） |
| `/mcp/servers` | GET/POST/DELETE | MCP 服务器列表/新增/删除 |
| `/mcp/servers/{name}/start` `/stop` `/status` | POST/GET | 服务器生命周期 |
| `/mcp/servers/{name}/tools` `/call` `/logs` | GET/POST | 工具列表/调用/日志 |
| `/mcp/tools/{name}/toggle` | POST | 工具启用/禁用 |
| `/local/tree/{doc_id}` | GET | 结构树文档树形结构 |
| `/local/clear` | DELETE | 清空数据库 |

---

## 15. 故障排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `torch.cuda.is_available()` 为 False | 装成了 CPU 版 torch（`+cpu`） | `poetry add torch==2.7.1 --source pytorch-cu128` |
| deepdoc 解析不加速 / onnxruntime 无 CUDA EP | onnxruntime 是 CPU 版或版本过高（1.27+ 用 CUDA 13 与 torch cu128 冲突） | 装 `onnxruntime-gpu==1.26.0`（见 3.2）；勿升 1.27+ |
| 上传 502 | embedding API 地址错误 | base_url 只写到 `/v1` |
| 上传后任务一直 PENDING | Redis 未启动 / Worker 未启动 | `docker ps` 确认 `rag-redis`；用 `--celery` 或单独起 Worker（见 3.5） |
| 双队列下任务卡在 PARSING 后不动 | 只起了 parse worker，`embedding_queue` 无人消费 | 两个队列各起一个 Worker（见 3.5） |
| 上传途中关程序，重开后任务「消失」 | 任务卡在运行态，未被正确标记 | 重启会自动标记为「失败：任务中断」，前端「最近入库任务」列表可重试（见 3.5 / 5.0） |
| 上传同名文件提示「正在上传」但实际没在传 | 上次中断的任务仍占用 `has_running` | 重启后自动清理遗留任务即可；或在前端历史列表手动重试/等待（见 5.0） |
| 任务重试后从中间步骤开始 | 步骤级重跑：前置步骤产物仍在，跳过重跑 | 属预期行为，非 bug（见 5.0） |
| 上传报「任务分派失败」 | Redis 连接失败 | 启动 Redis 容器：`docker run -d --name rag-redis -p 6379:6379 redis:7-alpine` |
| 同名文件上传报 409 | 该文件有任务正在入库 | 等任务完成或重试，再重新上传 |
| GPU OOM | 显存不足 | 降低 `retrieval_top_k` 或换 CPU |
| hybrid 检索偶发报错后自动恢复 | 入库写入 Milvus 期间并发查询的竞态 | 已内置降级：hybrid 失败自动降级 dense，不影响可用性 |
| 联网搜索崩溃 / 空结果 | 百度反爬 + GBK 编码崩溃 | 已修复 `websearch._log` 安全编码；Bing 兜底 |
| Agentic 评估卡死 | tool_llm 配 reasoning 模型未关思考 | 加 `disable_thinking: true` |
| 工具不生效 | MCP 服务器未启动 / 工具被禁用 | 用 `--mcp` 启动；检查 MCP 管理页工具开关 |
| 树导航只返回 2 条 | `min_evidences=2`「找到即停」设计 | 调大 `agentic.tree_nav.min_evidences` |
| 联网搜索返回无关内容 | 搜索引擎对冷门/新词无索引 | 属搜索引擎局限，非代码 bug |
| 答案还是旧风格（如猫娘） | 历史对话污染（历史里的旧回复被模仿） | 新建对话清空历史 |
| 前端页面打不开 / 白屏 | 没构建前端产物 `static/dist/` | 首次运行前 `cd frontend && npm install && npm run build`（见 3.6） |
| 改了前端代码但页面没变 | `server.py` 缓存了旧 `index.html`，且没重新构建 | `npm run build` 后重启主程序（见 3.6） |
| 数学公式显示成竖排 Unicode（`∫`） | 模型在「无法回答」兜底分支用 Unicode 拼公式，未按 `\(...\)` 输出 | 属工具调用被压制导致，修复后模型能正常回答即可输出 LaTeX（见 11.4） |
| Milvus 容器反复重启，`docker logs` 报 `panic: etcdserver: leader changed` | 非正常关机导致嵌入式 etcd 的 WAL/快照状态不一致，raft 恢复时崩溃（etcd 已知 bug） | 停容器 → 删除 `milvus/volumes/milvus/etcd` → 重启（etcd 全新初始化）→ 重新入库（见 16.7「换 embedding 必须重新入库」） |
| 检索结果 `ctx=0`（agentic 答案「无法回答」但无报错） | 两类原因：① embedding 服务断连，被 `retriever.py` 的 `except` 静默吞掉 → 0 证据提前停止；② Controller 首轮误判「信息已充分」直接 ANSWER | ① 检查 embedding 服务连通性；② 已在 `controller.py` 加硬约束（0 证据时强制先检索），升级到当前版本即可 |

---

## 16. RAG 评测（Ragas）

这一节介绍如何**量化评估**本系统的检索质量与生成质量——回答两个核心问题：「检索是不是把该找的都找出来了（召回率）？」「模型有没有一本正经地胡说八道（幻觉）？」。

### 16.1 为什么要评测、评什么

RAG 的质量可以拆成两条独立维度，分别对应两类典型问题：

| 关注点 | 典型症状 | 对应指标 |
|---|---|---|
| **召回精度** | 文档里明明有答案，却答不出来；或检索混入无关噪声 | Context Recall（召回率）、Context Precision（精度） |
| **推理准确率** | 答案「一本正经地胡说八道」 | Faithfulness（忠实度） |

本系统用 [Ragas](https://github.com/explodinggradients/ragas)（RAG Assessment）框架做自动化评测。Ragas 的核心思路是 **LLM-as-Judge（用大模型当裁判）**：把「答案是否忠实」「检索是否覆盖了答案所需信息」这类主观判断交给一个 LLM 打分，从而可规模化、可复现。

### 16.2 指标说明

| 指标 | 衡量什么 | 需要标准答案 | 对应问题 |
|---|---|---|---|
| **Faithfulness（忠实度）** | 答案有多少能被检索上下文支撑 | ❌ | **幻觉**：分数越低，编造越多 |
| **Context Recall（上下文召回率）** | 标准答案的关键信息，检索覆盖了多少 | ✅ | **漏检**：检索没找到关键片段 |
| **Context Precision（上下文精度）** | 检索片段里相关占比 + 相关片段是否排前 | ✅ | **噪声/排序**：混入无关、或相关被埋后面 |
| **Answer Relevancy（答案相关性）** | 答案是否切题、有无冗余 | ❌ | 答非所问、跑题 |
| **Answer Correctness（正确性）** | 答案与标准答案语义+事实一致度 | ✅ | 端到端对错 |

> **指标组合看**：只优化 Recall 会把 `top_k` 拉大 → Precision 崩 → 噪声多 → Faithfulness 反而降。所以要三者一起看。Ragas 的价值是**定位该改哪一环**（检索/重排/切分/生成提示词），而非追求绝对分数。

### 16.3 评测框架结构（eval/）

评测代码全部在 `eval/` 目录，**独立于主程序**，不污染运行环境：

```
eval/
├── config.py           # 组合矩阵 + 指标清单 + 模型配置读取（两个环境通用）
├── generate_dataset.py # 评测集生成（LLM 从文档生成 QA 对）
├── collect.py          # 组合遍历采集（跑 pipeline，采集 answer + contexts）
├── score.py            # Ragas 评分（5 指标）
├── report.py           # 汇总报告（组合 × 指标矩阵 + 维度对比）
├── run_full.py         # 一键编排（生成→采集→评分→报告，自动跨环境切换 + 断点续跑）
└── data/
    ├── dataset.json    # 评测集（20 题 QA 对）
    ├── runs/           # 采集结果 {combo_id}.json
    ├── scores/         # 评分结果 {combo_id}.json
    └── report.md       # 最终报告
```

**为什么拆两个环境**：主程序锁 `openai>=3.0.0`，而 ragas 依赖的 `instructor` 要求 `openai<3.0.0`，二者冲突。故采集在**主 poetry 环境**跑（复用 `server.py` / `rag_graph.py`），评分在**独立 venv**（`eval/.venv`）跑，中间用 JSON 文件交接。

### 16.4 组合矩阵（排列组合）

从代码确认，各维度对三种模式的作用范围不同：

| 维度 | 取值 | direct | rag | agentic |
|---|---|---|---|---|
| `mode` | direct / rag / agentic | — | — | — |
| `retrieval_mode` | vector / hybrid / tree | ✗ | ✓ | ✓ |
| `rewrite`（查询改写） | on / off | ✗ | ✓ | ✗ |
| `tool_calling`（工具调用） | on / off | ✗ | ✓ | ✗ |
| `websearch`（联网） | on / off | ✗ | ✓ | ✓ |

**组合总数 = 1（direct）+ 24（rag）+ 6（agentic）= 31 种**（见 `eval/config.py` 的 `COMBINATIONS`）。全量成本高，故内置「检索模式对比」子集（6 组，纯净对比 vector/hybrid/tree，关闭 rewrite/tool/websearch 避免干扰归因）：

```python
# eval/config.py → MODE_COMPARISON_IDS
["rag_vector_rw0_tc0_ws0", "rag_hybrid_rw0_tc0_ws0", "rag_tree_rw0_tc0_ws0",
 "agentic_vector_ws0", "agentic_hybrid_ws0", "agentic_tree_ws0"]
```

### 16.5 评测流程

**方式一（推荐）：一键编排 `run_full.py`**

一条命令完成「清空旧产物 → 生成 → 采集 → 评分 → 报告」全流程，自动切换采集/评分两个环境，并内置断点续跑：

```powershell
# 从头完整跑「检索模式对比」子集（6 组 × 20 题）
poetry run python eval/run_full.py

# 重新生成评测集后再完整跑
poetry run python eval/run_full.py --regenerate

# 断网/中断后续跑（跳过已完成题，自动补 ctx=0 空结果）
poetry run python eval/run_full.py --keep

# 只跑单个组合（先冒烟验证）
poetry run python eval/run_full.py --combo rag_hybrid_rw0_tc0_ws0
```

`run_full.py` 常用参数：`--subset all|mode-comparison`、`--combo <id>`、`--regenerate`、`--keep`（断点续跑）、`--skip-collect`（只评分+报告）、`--skip-score`（只采集）。

**方式二：分步手动执行**

```powershell
# ① 生成评测集（20 题：每篇 5 题 + 5 题跨文档）
poetry run python eval/generate_dataset.py

# ② 采集：跑 pipeline，采集 answer + contexts（断点续跑，每题落盘）
poetry run python eval/collect.py --subset mode-comparison

# ③ 评分：Ragas 5 指标（eval 独立 venv）
eval/.venv/Scripts/python eval/score.py --subset mode-comparison

# ④ 汇总报告
poetry run python eval/report.py
```

常用参数：

| 脚本 | 参数 | 说明 |
|---|---|---|
| `collect.py` | `--subset all\|mode-comparison` | 组合子集 |
| | `--combo <id>` | 只跑单个组合 |
| | `--limit N` | 每个组合只跑前 N 题（冒烟测试） |
| | `--refresh` | 忽略已有结果重跑 |
| | `--retry-empty` | 只重跑「检索结果为空（ctx=0 且无 error）」的题，其余跳过（断网恢复后补跑用） |
| `score.py` | `--subset` / `--combo` / `--limit` | 同上 |

报告 `eval/data/report.md` 输出：各模式「组合 × 指标」矩阵 + 按维度分组的对比表（rag/agentic 分开），直接回答「哪种检索模式召回精度更高、哪种幻觉更少」。

### 16.6 独立 venv 搭建（评分环境）

ragas 装不进主 poetry 环境（openai 版本冲突），需单独建 venv：

```powershell
python -m venv eval/.venv
eval/.venv/Scripts/python -m pip install ragas
# ragas 0.4.3 默认拉 langchain-community 0.4.x 会 import 失败，需降级：
eval/.venv/Scripts/python -m pip install "langchain-community<0.4.0"
```

### 16.7 关键配置与经验

**judge 模型必须用非 reasoning 模型**：ragas 的 `evaluate` 只接受旧版指标（`ragas.metrics._xxx`），judge 通过 `llm_factory` 走 instructor 结构化输出。若用 reasoning 模型（如 DeepSeek V4 Pro），thinking 无法禁用，`faithfulness` / `answer_correctness` 会因复杂 JSON 生成超时/不完整。本评测 judge 固定用 `deepseek-ai/DeepSeek-V3.2`（非 reasoning），配置见 `eval/config.py` 的 `JUDGE_MODEL`。

**两个必踩的坑**（已在 `eval/score.py` 修好）：

| 坑 | 现象 | 修复 |
|---|---|---|
| `max_tokens` 默认 1024 | faithfulness/correctness 的复杂 JSON 被截断 → `IncompleteOutputException` | `llm_factory(..., max_tokens=4096)` |
| `timeout` 默认 180s | answer_correctness 大 JSON 生成超时 | `RunConfig(timeout=600)` |

**换 embedding 必须重新入库**：现有知识库是用**特定 embedding 模型**向量化入库的，换模型（即使维度相同）会导致 query 向量与 chunk 向量不在同一语义空间，检索完全失效。换 embedding 前需先清空并重新入库（`db_service.clear_all()` 后重新 `ingest_file`）。本评测的 embedding 与主程序保持一致（`config/models.json` 的 `embedding`）。

**评测集质量决定上限**：Context Recall / Precision / Correctness 三个指标需要标准答案（ground_truth）。本项目用 LLM 从文档 chunk 自动生成 QA 对（`generate_dataset.py`），标准答案与文档内容强一致，但终究是「近似」，追求绝对分数意义有限——Ragas 更适合做**横向对比**（改切分/换 embedding/调检索模式前后）。

**采集时 agentic 的 `ctx=0` 要区分三种成因**（都会表现为「答案无法回答但无报错」）：

| 成因 | 特征 | 处理 |
|---|---|---|
| embedding 断连（静默降级） | `retriever.py` 的 `except` 吞掉异常 → 0 证据提前停止 | 检查 embedding 服务；网络恢复后 `--retry-empty` 补跑 |
| 真实漏检（vector 模式 query 英化偏移） | 同一题 hybrid 能检到、vector 检不到 | 属**真实评测发现**（vector 对中英偏移更敏感），保留 |
| 评测集幻觉题 | 知识库里根本没这内容（LLM 生成 ground_truth 时编造） | 属评测集质量问题，重生成时过滤覆盖不足的题 |

> 第一类（断连）是噪声，应补跑消除；后两类是**有价值的评测结论**，不应被当作错误剔除。

---

## 17. DeepSeek Harness 融合

### 17.1 是什么、为什么要融合

[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)（`dsh`，DeepSeek 开源的 Agent 运行时）定位是「**Model + Harness = Agent**」——一切皆插件（模型、工具、Agent Loop 都是插件），并**原生支持 MCP 协议**。

融合的意义：本系统的 RAG 检索能力已经封装成 MCP 工具（见第 11 章），因此可以把这些工具直接注册进 Harness，让 Harness 的**通用 Agent 循环**驱动本系统的知识库检索，形成「Harness Agent + RAG 检索」的融合体——用更成熟的 Agent 运行时替换/对比自研的 `agentic_rag`。

### 17.2 融合架构

```
┌───────────────────────────────────────────┐
│   DeepSeek Harness（Agent 前端）           │
│   任务拆解 → 工具编排 → 会话循环           │
│   模型: SiliconFlow DeepSeek（OpenAI 兼容）│
└───────────────────┬───────────────────────┘
                    │ MCP 协议（streamable-http）
                    ▼
┌───────────────────────────────────────────┐
│   mcp_service（本系统 RAG MCP 服务器）     │
│   search_knowledge_base / list_documents  │
│   get_knowledge_toc / web_search / ...    │
└───────────────────┬───────────────────────┘
                    ▼
        db_service + Milvus（知识库）
```

职责划分：Harness 负责「何时检索、检索什么、检索够不够、怎么汇总」，`mcp_service` 负责「具体检索」，两者通过 MCP 协议解耦。

### 17.3 接入配置

**前提 1：启动本系统的 MCP 服务**

```bash
poetry run python __main__.py --mcp   # 主程序 + MCP 一起启动，监听 http://127.0.0.1:8765/mcp
```

**前提 2：安装 Harness**（Node 环境）

```bash
npx @deepseek-ai/dsh
```

**① 模型统一到 SiliconFlow**（`~/.dsh/settings.yaml`）

Harness 支持「自定义 OpenAI 兼容端点」，无需 DeepSeek 官方 API：

```yaml
providers:
  siliconflow:
    apiKeyEnv: SILICONFLOW_API_KEY
    api: openai-completions
    baseURL: https://api.siliconflow.com/v1
    models:
      - id: deepseek-ai/DeepSeek-V3.2     # 建议先用非 reasoning 模型，兼容更稳
      - id: deepseek-ai/DeepSeek-V4-Pro-0813
```

**② 把 RAG 检索注册为 MCP 工具**（`~/.dsh/profiles/<profile>/cordis.patch.yml`）

`dsh-mcp-client` 是内核包（无需单独安装），用 `insert` 语法注册：

```yaml
- insert:
    - id: mcp-rag
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: rag
        transport: streamable-http
        url: http://127.0.0.1:8765/mcp
        toolCallTimeoutMs: 60000
        failOnStartupError: false
        reconnect:
          enabled: true
          initialDelayMs: 500
          maxDelayMs: 30000
          maxAttempts: 10
```

### 17.4 工具命名与验证

注册后，本系统的 6 个工具会以 `mcp__rag__<工具名>` 形式暴露给 agent：

```
mcp__rag__search_knowledge_base
mcp__rag__list_knowledge_documents
mcp__rag__get_knowledge_toc
mcp__rag__web_search
mcp__rag__calculate_pi
mcp__rag__calculate_expression
```

**验证配置**：`dsh web --dump-config` 应能看到 `mcp-rag` entry；重启 dsh 后在会话里问一个知识库问题，观察 agent 是否自动调用 `mcp__rag__search_knowledge_base`。

### 17.5 注意事项

| 事项 | 说明 |
|---|---|
| **Windows 系统代理** | 直连 `127.0.0.1:8765` 可能被系统代理劫持报 502（本系统 `manager.py` 已用 `NO_PROXY` 绕过）；若 Harness 侧也报 502，给 dsh 设 `NO_PROXY=127.0.0.1,localhost` |
| **reasoning 模型兼容** | DeepSeek-V4-Pro 的 `thinking` 会干扰 agent loop，建议先用 V3.2（非 reasoning） |
| **Windows 平台限制** | Harness 默认「编程 agent」组合（bash/PTY）不支持 Windows；RAG 检索场景不依赖 bash，可自定义轻量 profile（只挂 MCP 工具、去掉文件/bash 工具）在 Windows 跑 |
| **定位差异** | Harness 是**通用** Agent 运行时（面向写代码/文件），自研 `agentic_rag` 是**为 RAG 检索专门设计**（evidence 去重、coverage 判据等）。两者是「并行对比」关系，不是谁替代谁 |

---

## 附录：项目文件速查表

| 文件 | 一句话职责 |
|---|---|
| `__main__.py` | 主程序入口（`--mcp` 同步启停 MCP；`--celery` 同步启停 Celery Worker） |
| `server.py` | FastAPI 服务 + 所有 API + 服务 Vue3 前端产物 `static/dist/`（启动时清理遗留中断任务） |
| `rag_graph.py` | LangGraph 固定检索链路（含工具调用自主决策，见 11.4） |
| `frontend/` | 前端源码（Vue3 + Naive UI + Pinia + Vite），构建产物输出到 `static/dist/` |
| `tree_retrieval.py` | 纯树导航检索（通用模块，输出统一 dict） |
| `db_service.py` | 数据服务（embedding + Milvus 检索/入库/管理） |
| `celery_app.py` | Celery 应用 + 四步链式入库任务（parse/chunk/embed/index）+ 双队列路由 |
| `ingest_queue.py` | 异步入库任务队列（SQLite 任务表 + 状态机 + 步骤产物落盘 + 步骤级重跑 + 中断恢复） |
| `structure_resolver.py` | 结构归位（扁平元素 → 文档树） |
| `chunk_builder.py` | 结构树 → Retrieval Chunk 切分 |
| `tree_store.py` | 文档树持久化（SQLite）+ 结构查询原语 |
| `summarizer.py` | 章节摘要 + 文档主旨摘要（LLM） |
| `embedding.py` | Embedding + chunklet-py 父子块切分 |
| `milvus_store.py` | Milvus 封装（父子双 Collection + 混合检索） |
| `reranker.py` | Reranker（本地 HF / 远程 API） |
| `llm.py` | LLM 封装（ChatOpenAI / DoubaoLLM） |
| `llm_factory.py` | 模型工厂（按 kind 取模型，带缓存） |
| `dsml_read.py` | DSML 工具调用解析器 |
| `config_loader.py` / `store_config.py` | 配置加载 / 模型与数据库配置管理 |
| `common/text_utils.py` | 通用文本工具（健壮 JSON 解析 / CJK 检测 / 查询英化） |
| `agentic_rag/` | Agentic RAG 子包（多步检索循环） |
| `mcp_service/` | MCP 服务子包（工具 + 管理 + 桥接） |
| `eval/` | RAG 评测框架（Ragas，一键编排 run_full.py + 采集 + 评分 + 报告，见第 16 章） |
