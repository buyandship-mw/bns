from bns.utils.openai_client_azure import prompt_model

def main():
    prompt = "What is the capital of France?"
    response = prompt_model(prompt)
    print(f"Response from model: {response}")

if __name__ == "__main__":
    main()