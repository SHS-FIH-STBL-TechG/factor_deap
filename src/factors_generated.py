# src/factors_generated.py
"""
自动生成因子的加载与执行模块。

真正的因子定义（name/description/code）存放在同目录下的 factors_generated.json 里，
这个文件只负责：
  - 从 JSON 里读出因子列表
  - 提供 load_generated_factors() 供其他模块动态获取因子
  - 提供 add_generated_factors(df) 来在 df 上计算这些因子
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

# ===== 日志（可选，用你的 logging_utils，如果没有就 print） =====
try:
    from logging_utils import get_logger
    logger = get_logger("factors_generated")
except Exception:  # 兜底
    logger = None


def _log_info(msg: str):
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def _log_warning(msg: str):
    if logger is not None:
        logger.warning(msg)
    else:
        print("WARNING:", msg)


# JSON 文件路径：与本 py 文件同目录
FACTORS_JSON = Path(__file__).resolve().with_name("factors_generated.json")


def load_generated_factors() -> List[Dict[str, Any]]:
    """
    从 JSON 文件读取因子定义列表。
    JSON 结构是一个 list，每个元素是：
      {"name": "...", "description": "...", "code": "..."}
    """
    if not FACTORS_JSON.exists():
        _log_info(f"未找到 {FACTORS_JSON}，视为当前没有 AI 因子。")
        return []

    try:
        with FACTORS_JSON.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            _log_warning(f"{FACTORS_JSON} 内容不是 list，忽略。")
            return []
        return data
    except Exception as e:
        _log_warning(f"读取 {FACTORS_JSON} 失败：{e}")
        return []


# 为了兼容，保留一个“导入时的快照”，但后面基本不用它
GENERATED_FACTORS: List[Dict[str, Any]] = load_generated_factors()


def add_generated_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    把当前 JSON 里的所有 AI 因子都加到 df 上。
    注意：这里每次调用都会重新读取 JSON，保证用的是最新的因子定义。
    每个因子是一条 pandas 代码字符串，假设变量名为 df。
    """

    factors = load_generated_factors()
    if not factors:
        _log_info("当前没有可用的 AI 因子（factors_generated.json 为空或不存在）。")
        return df

    for fac in factors:
        name = fac.get("name")
        code = fac.get("code")
        if not name or not code:
            _log_warning(f"跳过无效因子定义：{fac}")
            continue

        try:
            # 给 factor 代码一个有 pd 的环境，df 通过局部变量传入
            exec(code, {"pd": pd}, {"df": df})
        except Exception as e:
            _log_warning(f"计算因子 {name} 失败: {e}")

    return df
