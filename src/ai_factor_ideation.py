# src/ai_factor_ideation.py
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from ai_clients.deepseek_client import ask_deepseek
from ai_clients.qwen_client import ask_qwen
from logging_utils import get_logger

logger = get_logger("ai_factor_ideation")

# 你当前 K 线 CSV 里的基础字段（和 factor_ga.py 里一致）
BASE_COLS: List[str] = [
    "交易日期", "开盘点位", "最高点位", "最低点位",
    "收盘价", "涨跌", "涨跌幅(%)", "开始日累计涨跌",
    "开始日累计涨跌幅", "成交量(万股)", "成交额(万元)", "持仓量",
]

# factor_ga.py 里手工写的基础因子名称
BASE_FACTORS: List[str] = [
    "因子_振幅",
    "因子_实体占振幅",
    "因子_成交额变化率",
    "因子_成交量变化率",
    "因子_当日涨跌幅",
]


def get_factors_json_path() -> Path:
    """返回 factors_generated.json 的路径（在 src 目录下）"""
    return Path(__file__).resolve().with_name("factors_generated.json")


def load_existing_generated_factors() -> List[Dict[str, Any]]:
    """
    从 factors_generated.json 中读取已有的 AI 因子列表。
    文件不存在时返回 []。
    """
    path = get_factors_json_path()
    if not path.exists():
        logger.info("未找到 %s，视为当前没有已有 AI 因子。", path)
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("%s 内容不是 list，忽略。", path)
            return []
        return data
    except Exception as e:
        logger.warning("读取 %s 失败：%s", path, e)
        return []


def write_factors_file(factors: List[Dict[str, Any]]) -> None:
    """
    把所有因子写入 factors_generated.json。
    注意：会整体覆盖原文件。
    """
    path = get_factors_json_path()
    with path.open("w", encoding="utf-8") as f:
        json.dump(factors, f, ensure_ascii=False, indent=2)
    logger.info("已更新 %s，当前自动生成因子数量：%d", path, len(factors))


def build_prompt(existing_factor_names: List[str]) -> str:
    """
    构造统一的提示词，要求模型输出 JSON。
    所有模型（DeepSeek / 千问）共用这一份 prompt。
    """
    base_cols_str = ", ".join(BASE_COLS)
    existing_str = ", ".join(existing_factor_names)

    prompt = f'''
你是一名熟悉中国A股市场的量化研究员，现在需要帮忙设计新的日频因子。

【已有数据列】
{base_cols_str}

【已经存在的因子名称】（这些名字不要再重复使用）
{existing_str}

数据是单只标的的日K线，用来预测“未来1日涨跌幅”。

要求你：
1. 设计 3 ~ 8 个新的因子，每个因子：
   - 必须使用 "因子_" 作为中文名称前缀，例如 "因子_5日收益动量"
   - 只能使用上述已有数据列构造
   - 只能使用当前或过去时点的信息，严禁用未来数据（不要用 shift(-1) 等）
2. 每个因子都给出简要含义描述，以及一行 pandas 代码。
   - 假设 DataFrame 变量名为 df
   - 因子赋值语句必须是下面这种形式之一：
     df["因子_xxx"] = ...
3. 输出格式必须是一个 **合法 JSON 数组**，例如：

[
  {{
    "name": "因子_示例1",
    "description": "这里写这个因子的含义。",
    "code": "df[\\"因子_示例1\\"] = df[\\"收盘价\\"].pct_change(periods=5)"
  }},
  {{
    "name": "因子_示例2",
    "description": "这里再写另一个因子的含义。",
    "code": "df[\\"因子_示例2\\"] = ..."
  }}
]

注意：
- 只能输出 JSON，不能在 JSON 外多写解释、注释或自然语言。
- JSON 数组里的每个对象必须同时包含 "name"、"description"、"code" 三个字段。
- 不要返回任何已经在【已经存在的因子名称】中的名称。
'''
    return prompt.strip()


def filter_invalid_factors(factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    用一个“虚拟 df”测试每个因子代码：
    - 能执行且生成同名列 → 认为是有效因子
    - 报错 / 不生成列 → 认为是“用不了的因子”，剔除
    """
    if not factors:
        return []

    n = 10
    data = {
        "交易日期": pd.date_range("2000-01-01", periods=n, freq="D"),
    }
    for col in BASE_COLS:
        if col == "交易日期":
            continue
        data[col] = [1.0] * n

    df_template = pd.DataFrame(data)

    valid = []
    removed = []

    for fac in factors:
        name = fac.get("name")
        code = fac.get("code")
        if not name or not code:
            removed.append((name, "缺少 name 或 code"))
            continue

        df_test = df_template.copy()
        try:
            # 这里要给 pd，否则 factor 里用 pd.xxx 会报错
            exec(code, {"pd": pd}, {"df": df_test})
        except Exception as e:
            removed.append((name, f"执行报错: {e}"))
            continue

        # 执行成功但没有生成同名列，也视为无效
        if name not in df_test.columns:
            removed.append((name, "执行成功但未生成同名列"))
            continue

        valid.append(fac)

    if removed:
        logger.info("因子自检：共 %d 个已有因子，剔除无效因子 %d 个：", len(factors), len(removed))
        for name, reason in removed:
            logger.info("  - 删除因子 %s，原因：%s", name, reason)

    return valid


def call_model(model: str, prompt: str) -> str:
    """
    根据模型名称调用对应的大模型。
    model 取值： "deepseek" / "qwen"
    """
    if model == "deepseek":
        system_prompt = "你是一名量化研究员，擅长基于行情数据设计因子，请严格按用户要求输出 JSON。"
        return ask_deepseek(system_prompt, prompt)
    elif model == "qwen":
        return ask_qwen(prompt)
    else:
        raise ValueError(f"未知模型名称: {model}")


def parse_factors_from_json(text: str) -> List[Dict[str, Any]]:
    """
    尝试从模型输出中解析出因子列表。
    如果 JSON 无法解析，返回空列表并打印 warning。
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("WARNING: 模型输出不是合法 JSON，解析失败：%s", e)
        logger.info("模型原始输出前 200 字符：%s", text[:200].replace("\n", " "))
        return []

    if not isinstance(data, list):
        logger.warning("WARNING: JSON 顶层不是数组，忽略。")
        return []

    factors: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        desc = item.get("description", "")
        code = item.get("code")
        if not name or not code:
            continue
        if not isinstance(name, str) or not isinstance(code, str):
            continue
        if not name.startswith("因子_"):
            continue
        factors.append({"name": name, "description": desc, "code": code})

    return factors


def main():
    # 1. 先拿到已有的“自动生成因子”
    existing_generated = load_existing_generated_factors()

    # 1.1 先做一遍自检，把用不了的因子删掉
    existing_generated = filter_invalid_factors(existing_generated)

    existing_generated_names = {
        f.get("name") for f in existing_generated if isinstance(f.get("name"), str)
    }

    # 2. 再加上手写的基础因子名，避免模型重复
    existing_names = set(BASE_FACTORS) | existing_generated_names

    # 3. 构造统一提示词
    prompt = build_prompt(sorted(existing_names))

    # 4. 依次调用 DeepSeek -> 千问（去掉 Gemini）
    models_in_order = [
        ("DeepSeek", "deepseek"),
        ("Qwen", "qwen"),
    ]

    new_factors: List[Dict[str, Any]] = []
    any_model_ok = False

    for label, model_id in models_in_order:
        logger.info(f"\n===== 正在调用 {label} 生成因子 =====")
        try:
            raw = call_model(model_id, prompt)
        except Exception as e:
            logger.info(f"WARNING: 调用 {label} 失败，跳过。详情：{e}")
            continue

        any_model_ok = True
        factors = parse_factors_from_json(raw)
        if not factors:
            logger.info(f"WARNING: {label} 返回的 JSON 没有解析出任何因子。")
            continue

        # 去重：已经有的名字不再加入
        added_count = 0
        for fac in factors:
            name = fac["name"]
            if name in existing_names:
                continue
            existing_names.add(name)
            new_factors.append(fac)
            added_count += 1

        logger.info(f"{label} 本次新增加因子数量：{added_count}")

    if not any_model_ok:
        raise RuntimeError("ERROR: 所有 AI 模型调用失败，无法生成新因子。")

    if not new_factors:
        logger.info("提示：AI 模型调用成功，但没有产生新的（未重复的）因子。")
        all_factors = existing_generated
    else:
        all_factors = existing_generated + new_factors

    # 5. 把合并后的因子写回 factors_generated.json
    write_factors_file(all_factors)


if __name__ == "__main__":
    main()
