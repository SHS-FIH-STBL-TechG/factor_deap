# src/ai_clients/qwen_client.py
"""
通义千问（DashScope）简单封装：
- 使用 text-generation 接口
- 模型：qwen-max（思考模型）
- 请求格式：
  {
    "model": "qwen-max",
    "input": {
      "messages": [
        {"role": "user", "content": "...prompt..."}
      ]
    },
    "parameters": {
      "result_format": "text"
    }
  }
- 返回结构参考（与你日志里的结构一致）：
  {
    "output": {
      "finish_reason": "stop",
      "text": "...模型输出..."
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
    这里要求模型以纯文本形式输出（output.text）。
    """
    api_key = _get_api_key()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # 注意：input 必须是 JSON 对象，而不是字符串
    payload = {
        "model": "qwen-max",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        },
        "parameters": {
            # 明确要求返回纯文本（对应你日志里的 output.text）
            "result_format": "text",
        },
    }

    resp = requests.post(
        DASHSCOPE_API_URL,
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False),
        timeout=60,
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Qwen HTTP {resp.status_code}: {resp.text[:200]}"
        )

    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Qwen 返回不是合法 JSON：{e}, 原始响应前 200 字符: {resp.text[:200]}")

    # 正常路径：output.text
    try:
        output = data.get("output", {})
        text = output.get("text")
        if not isinstance(text, str):
            raise KeyError("output.text 缺失或不是字符串")
        return text
    except Exception as e:
        logger.error("Qwen 返回结构异常，无法提取 output.text: %s, data=%s", e, data)
        raise RuntimeError(f"Qwen 返回结构异常，无法提取 output.text: {e}")
