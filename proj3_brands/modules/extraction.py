# extraction.py

import os
from dotenv import load_dotenv
from firecrawl import FirecrawlApp

load_dotenv()
API_KEY = os.getenv("FIRECRAWL_API_KEY")
if not API_KEY:
    raise EnvironmentError("FIRECRAWL_API_KEY not found")
APP = FirecrawlApp(api_key=API_KEY)

def parse_metadata(meta: dict) -> str:
    for key in ["og:title", "twitter:title", "title", "ogTitle", "name"]:
        v = meta.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return None

def fetch_metadata(url: str, timeout: int = 120000) -> dict:
    resp = APP.scrape_url(
        url=url,
        only_main_content=False,
        timeout=timeout
    )
    return resp.metadata

def extract_item_name(url: str) -> str:
    meta = fetch_metadata(url)
    if "error" in meta:
        raise RuntimeError(meta["error"])
    name = parse_metadata(meta)
    if not name:
        raise ValueError("No valid item name found in metadata")
    return name
