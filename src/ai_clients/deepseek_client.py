# src/ai_clients/deepseek_client.py
"""
DeepSeek API 封装：
- 使用 deepseek-reasoner（推理 / 思考模型）
- 暴露 ask_deepseek(system_prompt, user_prompt) -> str
"""

import os
import json
import requests
from typing import Optional

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-reasoner"  # 推理模型


class DeepSeekError(RuntimeError):
    pass


def ask_deepseek(system_prompt: str, user_prompt: str, timeout: int = 60) -> str:
    """
    调用 DeepSeek 推理模型，返回字符串形式的回复内容。

    参数:
        system_prompt: 系统提示（角色设定，比如“你是量化研究员...”）
        user_prompt:   用户问题（你的 JSON 因子生成提示词）
        timeout:       请求超时时间（秒）

    返回:
        模型回复的内容（string）
    """
    if not DEEPSEEK_API_KEY:
        raise DeepSeekError("DEEPSEEK_API_KEY 未配置，请在环境变量中设置 DeepSeek 的 API Key。")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # 你现在是让它严格按 JSON 输出，建议温度低一点
        "temperature": 0.2,
        # 如果你想要 OpenAI 风格的响应格式，可以加上：
        # "response_format": {"type": "text"},
    }

    resp = requests.post(
        DEEPSEEK_BASE_URL,
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=timeout,
    )

    if resp.status_code != 200:
        raise DeepSeekError(
            f"DeepSeek API 返回错误状态码 {resp.status_code}: {resp.text}"
        )

    try:
        data = resp.json()
    except Exception as e:
        raise DeepSeekError(f"解析 DeepSeek 返回的 JSON 失败: {e}, 文本={resp.text[:200]}")

    try:
        content: str = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise DeepSeekError(f"DeepSeek 返回结构异常，无法提取 content: {e}, data={data}")

    return content
