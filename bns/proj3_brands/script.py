# pipeline.py

import json
from modules.extraction import extract_item_name
from modules.prompting import build_prompt
from modules.model_client import prompt_model

def create_prompt_from_url(url: str) -> str:
    """
    Step 1: go from URL → prompt.
    """
    try:
        item_name = extract_item_name(url)
        return build_prompt(item_name)
    except Exception:
        # fallback to prompting on the URL itself
        return build_prompt(url)

def get_brand_from_prompt(prompt: str) -> dict:
    """
    Step 2: go from prompt → parsed brand dict.
    """
    raw = prompt_model(prompt)
    return json.loads(raw)

if __name__ == "__main__":
    test_url = "https://www.adidas.com.hk/en/stadt-shoes/JI1886.html"
    print("Testing with URL:", test_url)
    print()
    # Step 1
    prompt = create_prompt_from_url(test_url)
    print("Built prompt:\n", prompt)
    # Step 2
    brand_info = get_brand_from_prompt(prompt)
    print("Extracted brand:", brand_info.get("name", "No brand found"))