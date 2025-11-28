# src/ai_clients/qwen_client.py
"""
通义千问（DashScope）API 封装：
- 使用 qwen-max（高阶 / 推理较强的模型）
- 暴露 ask_qwen(user_prompt) -> str
"""

import os
import json
import requests
from typing import Optional

# 阿里云百炼的 API Key，一般是 DASHSCOPE_API_KEY
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
QWEN_MODEL = "qwen-max"  # 高阶 / 推理较强
QWEN_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


class QwenError(RuntimeError):
    pass


def ask_qwen(user_prompt: str, timeout: int = 60) -> str:
    """
    调用通义千问 qwen-max 模型，返回字符串形式的回复内容。

    参数:
        user_prompt:  你的完整提示词（这里直接用 ai_factor_ideation 里构造的 JSON 提示词）
        timeout:      请求超时时间（秒）

    返回:
        模型回复的内容（string）
    """
    if not QWEN_API_KEY:
        raise QwenError("未找到通义千问 API Key，请在环境变量 DASHSCOPE_API_KEY 或 QWEN_API_KEY 中配置。")

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "disable",
    }

    # 这里统一加一个 system 角色，告诉它你是量化研究员 & 严格输出 JSON
    system_prompt = (
        "你是一名擅长中国A股日频量化因子设计的研究员，必须严格按照用户要求返回 JSON 格式。"
    )

    payload = {
        "model": QWEN_MODEL,
        "input": {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        },
        "parameters": {
            # 结果格式设为 text，方便直接拿 content
            "result_format": "text",
            "temperature": 0.2,
        },
    }

    resp = requests.post(
        QWEN_URL,
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=timeout,
    )

    if resp.status_code != 200:
        raise QwenError(
            f"Qwen API 返回错误状态码 {resp.status_code}: {resp.text}"
        )

    try:
        data = resp.json()
    except Exception as e:
        raise QwenError(f"解析 Qwen 返回的 JSON 失败: {e}, 文本={resp.text[:200]}")

    # DashScope 返回结构一般是 output -> choices -> [ { message: { content: ... } } ]
    try:
        content: str = data["output"]["choices"][0]["message"]["content"]
    except Exception as e:
        raise QwenError(f"Qwen 返回结构异常，无法提取 content: {e}, data={data}")

    return content
