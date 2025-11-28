# src/ai_clients/qwen_client.py
"""
通义千问（DashScope）简单封装：
- 使用 text-generation 接口
- 模型：qwen-max（思考模型）
- 返回的数据结构形如：
  {
    "output": {
      "text": "...",
      "finish_reason": "stop"
    },
    "usage": {...},
    "request_id": "..."
  }
"""

import os
import json
import requests

from logging_utils import get_logger

logger = get_logger("qwen_client")

DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


def _get_api_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not key:
        raise RuntimeError("未找到通义千问 API Key，请设置环境变量 DASHSCOPE_API_KEY 或 QWEN_API_KEY")
    return key


def ask_qwen(prompt: str) -> str:
    """
    向通义千问发送 prompt，返回字符串形式的回复。
    要求模型以纯文本形式输出（这里我们再去解析 JSON）。
    """
    api_key = _get_api_key()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": "qwen-max",  # 思考能力较强的大模型
        "input": prompt,
        "parameters": {
            # 明确要求返回纯文本（对应你日志里的 output.text）
            "result_format": "text",
        },
    }

    resp = requests.post(DASHSCOPE_API_URL, headers=headers, data=json.dumps(payload), timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Qwen HTTP {resp.status_code}: {resp.text[:200]}"
        )

    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Qwen 返回不是合法 JSON：{e}, 原始响应前 200 字符: {resp.text[:200]}")

    # 优先走 output.text（与你日志中的结构一致）
    try:
        output = data.get("output", {})
        text = output.get("text")
        if not isinstance(text, str):
            raise KeyError("output.text 缺失或不是字符串")
        return text
    except Exception as e:
        # 兜底：如果未来 DashScope 又改结构，这里打印一份完整 data 方便你排查
        logger.error("Qwen 返回结构异常，无法提取 output.text: %s, data=%s", e, data)
        raise RuntimeError(f"Qwen 返回结构异常，无法提取 output.text: {e}")
