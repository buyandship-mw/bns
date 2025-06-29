# model_client.py

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT") + "openai/v1/",
    default_query={"api-version": "preview"},
)

def prompt_model(prompt: str) -> str:
    try:
        resp = _client.responses.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            input=prompt
        )
        return resp.output[0].content[0].text
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}")
