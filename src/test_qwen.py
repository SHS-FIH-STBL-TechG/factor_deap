from ai_clients.qwen_client import ask_qwen

if __name__ == "__main__":
    answer = ask_qwen("简单解释一下什么是股票的换手率。")
    print(answer)
