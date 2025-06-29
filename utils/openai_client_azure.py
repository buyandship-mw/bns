import os
import configparser
from openai import AzureOpenAI

# Read Azure OpenAI config from config.ini
config = configparser.ConfigParser()
config_file = os.path.join(os.path.dirname(__file__), "../../config.ini")
config.read(config_file)

api_key = config['azure_openai']['api_key']
endpoint = config['azure_openai']['endpoint']
deployment = config['azure_openai']['deployment']
api_version = config['azure_openai']['api_version']

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint
)

def prompt_model(prompt, temp=1.0):
    """
    Sends a prompt to the Azure OpenAI model and returns the response.
    """
    response = client.chat.completions.create(
        model=deployment,  # this must match the *deployment name* in Azure
        temperature=temp,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    text = prompt_model("Say hello world back to me.")
    print(text)