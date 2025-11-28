# src/logging_utils.py
"""
全局统一日志配置：

- 所有模块通过 get_logger(name) 获取 logger
- 所有日志写入同一个文件 logs/log_YYMMDDHHMMSS.log
- 同时输出到控制台
- 日志格式：时间 [级别] [logger名] [函数名] 消息
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# 是否已经初始化过 logging
_LOGGING_INITIALIZED = False
_LOG_FILE_PATH: Optional[Path] = None


def _init_logging():
    """
    初始化全局 logging 配置：
    - 创建 logs 目录
    - 根据当前时间生成一个 log_YYMMDDHHMMSS.log 文件
    - 给 root logger 挂上 FileHandler 和 StreamHandler
    """
    global _LOGGING_INITIALIZED, _LOG_FILE_PATH
    if _LOGGING_INITIALIZED:
        return

    # 1. 日志目录：项目根目录下的 logs/
    # 假设 logging_utils.py 在 src/ 目录下
    root_dir = Path(__file__).resolve().parent.parent
    log_dir = root_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # 2. 根据当前时间生成文件名：log_YYMMDDHHMMSS.log
    ts = datetime.now().strftime("%y%m%d%H%M%S")
    _LOG_FILE_PATH = log_dir / f"log_{ts}.log"

    # 3. 配置 root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 日志格式：时间 [级别] [logger名] [函数名] 消息
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] [%(funcName)s] %(message)s"
    )

    # 文件输出
    fh = logging.FileHandler(_LOG_FILE_PATH, encoding="utf-8")
    fh.setFormatter(fmt)

    # 控制台输出
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    root_logger.addHandler(fh)
    root_logger.addHandler(ch)

    _LOGGING_INITIALIZED = True


def get_logger(name: str = "root") -> logging.Logger:
    """
    获取一个带名字的 logger。
    所有 logger 共用同一个 root 配置和同一个日志文件。
    """
    _init_logging()
    logger = logging.getLogger(name)
    # 使用 root 的 handler，不再单独加 handler
    logger.propagate = True
    return logger


def get_log_file_path() -> Optional[Path]:
    """
    返回当前使用的日志文件路径（Path 对象），
    方便你在程序里打印或写进报告。
    """
    _init_logging()
    return _LOG_FILE_PATH
