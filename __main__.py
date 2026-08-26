"""
================================================================================
RAG 知识库系统 — 程序入口
================================================================================

启动方式：
  python __main__.py                        # 默认 127.0.0.1:8000
  python __main__.py --mcp                  # 同步启动/关闭 MCP 服务器
  python __main__.py --celery               # 同步启动/关闭 Celery Worker（异步入库）
  python __main__.py --port 8080 --host 0.0.0.0

说明：
  检索/入库/对话历史为本地直连 Milvus，不依赖 MCP 服务器。
  所有工具注册在 MCP 服务器上，主程序调用工具走 MCP。
  用 --mcp 启动时 MCP 服务器随主程序同步启停；也可单独运行：
      python -m mcp_service --host 127.0.0.1 --port 8765
  异步入库需要 Celery Worker 消费 Redis 队列：加 --celery 随主程序同步启停
  （Worker 日志写入 data/logs/celery_worker.log）；也可单独运行：
      poetry run celery -A celery_app worker --pool=solo

服务逻辑在 server.py（仅定义 FastAPI app，不直接运行），
本文件是唯一启动入口。
================================================================================
"""

import argparse
import os
import signal
import subprocess
import sys
import threading

# ---------------------------------------------------------------------------
# 关键修复：torch 必须作为本程序的第一个 import（最浅调用栈）完整加载。
# 否则后续 langchain_openai / transformers 触发的 torch 二次 import 会因
# torch._library.utils.get_source 在深层调用栈下 inspect 崩溃而报错
# （"partially initialized module 'torch'" / "Only a single TORCH_LIBRARY"）。
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401
except ImportError:
    pass

import uvicorn

from server import app
from config_loader import config


def _kill_tree(proc: subprocess.Popen) -> None:
    """终止进程及其整个进程树。

    Windows 上 Popen.terminate() 只调用 TerminateProcess 杀直接子进程本身，
    不杀它 spawn 的子进程（celery worker / MCP 会再 spawn billiard 进程池等），
    导致孙进程变成孤儿进程残留。这里用 taskkill /T /F 杀整棵进程树；
    Unix 用 terminate → 超时 kill。
    """
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return  # 已退出
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True, timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    except Exception:
        pass


class _ForceExitServer(uvicorn.Server):
    """自定义 Server：禁用 uvicorn 内置的 graceful shutdown 信号处理。

    uvicorn 默认在收到 Ctrl+C（SIGINT）时进入 graceful shutdown，会打印
    "Waiting for connections to close" 并无限等待 SSE 长连接（前端 fetch 流）
    关闭，导致进程卡死（此时 lifespan 的 shutdown 段与 finally 里的
    os._exit(0) 都执行不到）。

    这里禁用其内置 handler，改由 main() 里自定义的 handler 在收到信号时
    直接 os._exit(0) 强杀，绕过 graceful shutdown 的等待。
    """

    def install_signal_handlers(self) -> None:
        # 故意空实现：不让 uvicorn 覆盖我们的信号处理器
        pass


def main():
    _svr_cfg = config["server"]
    p = argparse.ArgumentParser(description="RAG HTTP 服务")
    p.add_argument("--port", type=int, default=_svr_cfg["port"])
    p.add_argument("--host", type=str, default=_svr_cfg["host"])
    p.add_argument("--mcp", action="store_true",
                   help="启动主程序时同步启动 MCP 服务器，退出时同步关闭")
    p.add_argument("--celery", action="store_true",
                   help="启动主程序时同步启动 Celery Worker（异步入库），退出时同步关闭")
    args = p.parse_args()

    mcp_enabled = args.mcp
    celery_enabled = args.celery

    _celery_proc = None
    _celery_log = None

    def _start_celery():
        """随主程序启动 Celery Worker（子进程，消费 Redis 队列执行异步入库）。"""
        nonlocal _celery_proc, _celery_log
        if not celery_enabled:
            return
        try:
            # 从 config 读 worker 池类型与并发数（默认 threads 并发 4，多文件并行入库）
            pool, concurrency = "threads", 4
            try:
                from config_loader import cfg
                pool = str(cfg("ingest.worker_pool", "threads")).strip().lower()
                concurrency = int(cfg("ingest.worker_concurrency", 4))
            except Exception:
                pass

            # 日志写入 data/logs/celery_worker.log（不污染主程序终端，便于排障）
            base = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(base, "data", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "celery_worker.log")
            _celery_log = open(log_path, "w", encoding="utf-8")

            # 复用当前 Python 环境（poetry），无需再套 poetry run
            cmd = [sys.executable, "-m", "celery", "-A", "celery_app", "worker",
                   "--loglevel=info"]
            if pool == "solo":
                cmd += ["--pool=solo"]  # 单进程串行（多 PDF 深度解析场景）
            else:
                cmd += ["--pool=threads", f"--concurrency={concurrency}"]  # 并发（默认）

            # CREATE_NEW_PROCESS_GROUP：让 worker 拥有独立进程组，配合 _kill_tree
            # 用 taskkill /T 整树杀干净（否则 worker spawn 的子进程会残留）。
            _flags = 0
            if os.name == "nt":
                _flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            proc = subprocess.Popen(
                cmd,
                cwd=base,
                stdout=_celery_log,
                stderr=subprocess.STDOUT,
                creationflags=_flags,
            )
            _celery_proc = proc
            print(f"[Celery] Worker 已启动 (pid={proc.pid}, pool={pool}, "
                  f"concurrency={concurrency}, 日志: {log_path})")

            # 后台监控退出：worker 意外退出时提示（否则用户会误以为还在消费队列）
            def _monitor():
                nonlocal _celery_proc, _celery_log
                code = proc.wait()
                print(f"[Celery] Worker 进程退出 (code={code})，请检查日志或 Redis 是否正常")
                _celery_proc = None
                try:
                    _celery_log.close()
                except Exception:
                    pass
                _celery_log = None

            threading.Thread(target=_monitor, daemon=True).start()
        except Exception as e:
            print(f"[Celery] Worker 启动失败: {e}")

    def _stop_celery():
        """随主程序关闭 Celery Worker（杀进程树），并标记被中断的入库任务。"""
        nonlocal _celery_proc, _celery_log
        if _celery_proc is None:
            return
        try:
            _kill_tree(_celery_proc)
            print("[Celery] Worker 已停止")
        except Exception as e:
            print(f"[Celery] Worker 关闭失败: {e}")
        finally:
            _celery_proc = None
            if _celery_log is not None:
                try:
                    _celery_log.close()
                except Exception:
                    pass
                _celery_log = None
            # Worker 退出后，把「运行中」任务标记为中断（否则状态永久卡在中间态）。
            # 在 worker 完全退出后执行，避免 worker 仍在写状态造成竞态。
            try:
                import ingest_queue
                n = ingest_queue.interrupt_running()
                if n:
                    print(f"[Celery] 有 {n} 个入库任务因服务关闭而中断（重启后可在前端重试）")
            except Exception as e:
                print(f"[Celery] 标记中断任务失败: {e}")

    def _start_mcp():
        """随主程序启动 MCP 服务器（子进程）。"""
        if not mcp_enabled:
            return
        try:
            from mcp_service import manager
            res = manager.start_server("RAG-Service")
            print(f"[MCP] {res.get('message', '')}")
        except Exception as e:
            print(f"[MCP] 启动失败: {e}")

    def _stop_mcp():
        """随主程序关闭 MCP 服务器（终止子进程）。"""
        if not mcp_enabled:
            return
        try:
            from mcp_service import manager
            manager.stop_server("RAG-Service")
        except Exception as e:
            print(f"[MCP] 关闭失败: {e}")

    def _force_exit(signum, frame):
        """收到 Ctrl+C / 终止信号时立即强杀，绕过 uvicorn graceful shutdown。

        uvicorn 的 graceful shutdown 会等待 SSE 长连接关闭而卡死，
        所以这里先关闭 MCP 服务器与 Celery Worker 再直接 os._exit(0)。
        每个 stop 单独 try 保护：任一失败都不阻塞最终 os._exit(0) 强杀。
        """
        try:
            _stop_celery()
        except Exception:
            pass
        try:
            _stop_mcp()
        except Exception:
            pass
        os._exit(0)

    # 注册退出处理器，覆盖所有「关闭程序」的方式。
    # Windows：用 SetConsoleCtrlHandler 统一处理 Ctrl+C(0) / Ctrl+Break(1) /
    #   关窗口(2) / 注销(5) / 关机(6)。关键：Python 的 signal 模块不处理
    #   CTRL_CLOSE_EVENT（点终端 X 关窗口），若用户这样关闭，主进程被强杀而
    #   _force_exit 不执行；celery/MCP 子进程因用 CREATE_NO_WINDOW（无 console）
    #   启动、不会随 console 关闭而终止，从而变成孤儿进程残留。这里用底层
    #   handler 兜底，确保任何关闭方式都会执行清理（taskkill 杀进程树 + os._exit）。
    if os.name == "nt":
        import ctypes

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
        def _console_handler(ctrl_type: int) -> bool:
            # CTRL_C_EVENT=0, CTRL_BREAK_EVENT=1, CTRL_CLOSE_EVENT=2,
            # CTRL_LOGOFF_EVENT=5, CTRL_SHUTDOWN_EVENT=6
            if ctrl_type in (0, 1, 2, 5, 6):
                try:
                    _force_exit(ctrl_type, None)
                except Exception:
                    pass
                os._exit(0)  # 兜底：_force_exit 若意外返回，仍强制退出
            return True

        # 保存引用，防止 handler 被垃圾回收导致回调失效
        _console_handler_ref = _console_handler
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_console_handler_ref, True)
    else:
        # Unix：用 signal（SIGINT=Ctrl+C，SIGTERM=kill）
        signal.signal(signal.SIGINT, _force_exit)
        try:
            signal.signal(signal.SIGTERM, _force_exit)
        except (AttributeError, ValueError):
            pass

    # 随主程序启动 MCP 服务器（--mcp 启用时）
    _start_mcp()
    # 随主程序启动 Celery Worker（--celery 启用时）
    _start_celery()

    print(f"启动: http://{args.host}:{args.port}")

    config_uvicorn = uvicorn.Config(app, host=args.host, port=args.port)
    server = _ForceExitServer(config_uvicorn)
    try:
        server.run()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # 兜底强制退出：正常情况下 _force_exit 已 os._exit(0) 直接结束，
        # 这里仅防御 server.run() 以其它方式返回的情况。
        _stop_celery()
        _stop_mcp()
        os._exit(0)


if __name__ == "__main__":
    main()
