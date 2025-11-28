# src/ai_clients/gemini_client.py
import os
from google import genai


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("请先在环境变量中设置 GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


def ask_gemini(prompt: str,
               model: str = "gemini-2.0-flash") -> str:
    """
    调用 Gemini 文本模型，返回回复文本。
    model 名称可以根据你的账号实际支持的型号调整。
    """
    client = get_gemini_client()
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return resp.text
