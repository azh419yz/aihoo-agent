"""日志配置（按天切分，2026-09-23 起）

独立成模块的原因：
1. `app/main.py` 在模块级挂载 file handler —— 任何 `import app.main`（含 pytest 收集）
   都会产生「创建 logs/agent.log + 注册 root handler」的副作用，进而让**测试期间的日志
   写进生产日志文件**。把配置抽到这里，测试只 import 本模块即可，无副作用。
2. 生产与测试共用同一份实现（`daily_namer` / `build_daily_file_handler`），避免配置漂移。

滚动策略：
- 当天写入 `logs/agent.log`；跨天后历史归档为 `logs/agent-YYYY-MM-DD.log`（= **内容所属日期**）。
- 切换时机是「午夜后第一条日志到达时」，静默期不产生空文件。
- 保留 `LOG_RETAIN_DAYS` 天，超期由 handler 的 backupCount 自动清理
  （Python 3.12+ 的 `getFilesToDelete` 会用 namer 反推比对，自定义命名下清理仍然生效）。
- 注意：`TimedRotatingFileHandler.__init__` 在目标文件已存在时，用**文件 mtime** 反算切分点。
  所以服务停机跨天再启动时，旧内容会立刻被归档成它 mtime 所属日期的文件（行为正确，
  但排查时别把「一启动就多出一个历史文件」误判为异常）。
"""

from __future__ import annotations

import logging
import os
import re
from logging.handlers import TimedRotatingFileHandler

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

#: 历史日志保留天数
LOG_RETAIN_DAYS = 30


def daily_namer(default_name: str) -> str:
    """历史日志命名：logs/agent.log.2026-09-22 → logs/agent-2026-09-22.log

    传给 `TimedRotatingFileHandler.namer`。不匹配日期后缀的名字（如旧的大小滚动产物
    `agent.log.1`）原样返回。
    """
    return re.sub(r"^(.*/agent)\.log\.(\d{4}-\d{2}-\d{2})$", r"\1-\2.log", default_name)


def build_daily_file_handler(
    log_dir: str,
    level: int = logging.INFO,
    retain_days: int = LOG_RETAIN_DAYS,
    filename: str = "agent.log",
) -> TimedRotatingFileHandler:
    """构造「按天切分」的日志 handler（本地时区 00:00 滚动）"""
    os.makedirs(log_dir, exist_ok=True)
    handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, filename),
        when="midnight",
        interval=1,
        backupCount=retain_days,
        encoding="utf-8",
        utc=False,
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.namer = daily_namer
    return handler
