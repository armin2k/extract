from dotenv import set_key

def setup():
    print("Welcome! Let's configure your API keys:")
    
    deepseek_key = input("Enter DeepSeek API key (leave empty to skip): ")
    chatgpt_key = input("Enter ChatGPT API key (leave empty to skip): ")
    
    set_key(".env", "DEEPSEEK_API_KEY", deepseek_key)
    set_key(".env", "OPENAI_API_KEY", chatgpt_key)
    
    print("\nConfiguration saved! Now run: python financial_analyzer.py")

if __name__ == "__main__":
    setup()