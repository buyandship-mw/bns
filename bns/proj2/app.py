import json
from itertools import islice
import concurrent.futures
import os
import time
from typing import Optional

from bns.utils.openai_client_azure import prompt_model # Your actual import

# --- Constants ---
PROMPT_TEMPLATE = """You are an expert in product cataloging. Your task is to extract the primary brand name and the specific item type from product names.

Key Guidelines for Brand Extraction:
1.  **Identify the Core Brand:** Focus on the main recognizable brand.
2.  **Normalize Variations:**
    * Convert different spellings or accented characters to a common, unaccented form (e.g., "Pokémon" should become "Pokemon").
    * Attribute sub-brands, store-specific versions (like "Pokemon Center", "Pokemon Center Yokohama"), or product lines directly related to a core brand back to that core brand. For example, if the item is clearly a "Pokemon" product, the brand should be "Pokemon", even if "Pokemon Center" is mentioned.
3.  **Null for Unclear Brands:** If a distinct product brand cannot be identified, use null for "brand_name".

Return strictly the JSON result in the format specified by the examples.

--- EXAMPLES ---
Product name: Colgate Total Whitening Toothpaste 120g
Result: {{"brand_name": "Colgate", "item_type": "Toothpaste"}}
Product name: Extra Virgin Olive Oil 500ml
Result: {{"brand_name": null, "item_type": "Olive Oil"}}
Product name: 0308069710162006000999
Result: {{"brand_name": null, "item_type": null}}
Product name: Pokémon Surging Sparks Elite Trainer Box [Pokémon Center Exclusive]
Result: {{"brand_name": "Pokemon", "item_type": "Elite Trainer Box"}}
Product name: Sapporo's Pikachu 005/SM-P Pokemon Center 2016 - Japanese - PSA 10 Gem Mint
Result: {{"brand_name": "Pokemon", "item_type": "Pokemon Card"}}
Product name: BGS 6 YOKOHAMAS'S PIKACHU  | POKEMON CENTER YOKOHAMA SPECIAL BOX 281/SM-P
Result: {{"brand_name": "Pokemon", "item_type": "Special Box"}}
Product name: Nike Air Max 90
Result: {{"brand_name": "Nike", "item_type": "Shoes"}}
Product name: Apple iPhone 15 Pro Max
Result: {{"brand_name": "Apple", "item_type": "Smartphone"}}
--- END EXAMPLES ---

Return strictly the JSON result for the below product name.
Product name: {name}
"""
DEFAULT_ITEMS_FILEPATH = 'bns/proj2/data/items.json'
DEFAULT_OUTPUT_FILEPATH = 'bns/proj2/data/branded_items.jsonl'

# --- Data Loading ---
def load_items_from_jsonl(filepath: str):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    item_name = item.get('item_name', '').strip()
                    item_ct = item.get('item_ct')
                    if not item_name or item_ct is None:
                        # print(f"Warning: Skipping line {line_number} in '{filepath}' due to missing 'item_name' or 'item_ct': {line}")
                        continue
                    yield (item_name, item['item_ct'])
                except json.JSONDecodeError: # e
                    # print(f"Warning: Skipping line {line_number} in '{filepath}' due to JSON decoding error: {e}. Line content: '{line}'")
                    continue
                except KeyError: # e
                    # print(f"Warning: Skipping line {line_number} in '{filepath}' due to missing key {e}. Line content: '{line}'")
                    continue
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        raise

# --- Prompt Engineering ---
def format_llm_prompt(item_name: str, template: str = PROMPT_TEMPLATE) -> str:
    return template.format(name=item_name)

# --- LLM Interaction and Response Parsing ---
def get_item_labels_from_llm(item_name: str, prompt_template: str = PROMPT_TEMPLATE):
    prompt = format_llm_prompt(item_name, prompt_template)
    try:
        raw_response = prompt_model(prompt)
    except Exception as e:
        print(f"Error during LLM call for item '{item_name}': {e}")
        return None, None

    if raw_response is None:
        print(f"Error during LLM call for item '{item_name}': Received None response.")
        return None, None
    else:
        cleaned_response_str = raw_response.strip()

    if cleaned_response_str.startswith("```json") and cleaned_response_str.endswith("```"):
        json_str_to_parse = cleaned_response_str[len("```json") : -len("```")].strip()
    elif cleaned_response_str.startswith("```") and cleaned_response_str.endswith("```"):
        json_str_to_parse = cleaned_response_str[len("```") : -len("```")].strip()
    else:
        json_str_to_parse = cleaned_response_str

    try:
        if not json_str_to_parse:
            raise json.JSONDecodeError("Extracted JSON string is empty after stripping fences.", cleaned_response_str, 0)
        label_data = json.loads(json_str_to_parse)
        brand_name = label_data.get("brand_name")
        item_type = label_data.get("item_type")
        if brand_name is not None:
            brand_name = brand_name.title()
        if item_type is not None:
            item_type = item_type.title()
        return brand_name, item_type
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM JSON response for item '{item_name}': {e}. "
              f"Attempted to parse (first 200 chars): '{json_str_to_parse[:200]}...' "
              f"Raw response (first 200 chars): '{raw_response[:200]}...'")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred while parsing LLM response for item '{item_name}': {e}")
        return None, None

# --- Helper function for concurrent processing ---
def process_single_item_task(item_data, prompt_template_str):
    name, ct = item_data
    brand_name, item_type = get_item_labels_from_llm(name, prompt_template_str)

    if brand_name is None and item_type is None:
        print(f"Skipping (in thread): Name='{name}' as both brand_name and item_type are null.")
        return None

    processed_item = {
        "item_name": name,
        "brand_name": brand_name,
        "item_type": item_type,
        "item_ct": ct
    }
    print(f"Processed (in thread): Name='{name}', Brand='{brand_name}', Type='{item_type}', Count={ct}")
    return processed_item

# --- Data Processing (Generator) - Modified for Concurrency & Skipping ---
def generate_processed_items_concurrently(items_iterable,
                                          prompt_template_str: str = PROMPT_TEMPLATE,
                                          max_workers: int = 1,
                                          already_processed_names: Optional[set] = None):
    if already_processed_names is None:
        already_processed_names = set()

    futures = []
    skipped_count = 0
    submitted_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for item_data in items_iterable:
            item_name_to_check = item_data[0]
            if item_name_to_check in already_processed_names:
                skipped_count += 1
                continue
            futures.append(executor.submit(process_single_item_task, item_data, prompt_template_str))
            submitted_count +=1
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} items found in the existing output file.")
        if submitted_count == 0 and skipped_count > 0:
             print("No new items to process; all were found as already processed.")
        elif submitted_count == 0 and skipped_count == 0:
            print("No items were submitted for processing (input might be empty or fully processed).")


        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    yield result
            except Exception as exc:
                # This error reporting might need item context if available
                # For now, it refers to a generic item processing failure.
                print(f'An item processing task generated an exception: {exc}')


# --- Data Saving (JSONL) - Modified for Appending ---
def save_results_to_jsonl(processed_items_generator, output_path: str) -> int:
    count_saved_this_run = 0
    try:
        # Open in append mode ('a')
        with open(output_path, 'a', encoding='utf-8') as f:
            for item in processed_items_generator:
                try:
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(json_line + '\n')
                    count_saved_this_run += 1
                except TypeError as e:
                    print(f"Error serializing item to JSON: {e}. Item: {item}")
        if count_saved_this_run > 0:
            print(f"Successfully appended {count_saved_this_run} new labeled items to {output_path}")
        elif os.path.exists(output_path): # File exists but nothing new was appended
             print(f"No new items were appended to {output_path} in this run.")

        return count_saved_this_run
    except IOError as e:
        print(f"Error writing to output file {output_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving results to {output_path}: {e}")
    return count_saved_this_run

# --- Main Execution - Modified for Resumability ---
def main(items_filepath: str = DEFAULT_ITEMS_FILEPATH,
         output_filepath: str = DEFAULT_OUTPUT_FILEPATH,
         max_items: Optional[int] = None,
         num_workers: int = 1):
    print(f"Starting item labeling process...")
    print(f"Input file: {items_filepath}")
    print(f"Output file (appending): {output_filepath}")
    if num_workers:
        print(f"Requested number of worker threads: {num_workers}")

    start_time = time.time()

    # Load names of already processed items to avoid duplicates
    processed_item_names = set()
    if os.path.exists(output_filepath):
        print(f"Checking existing output file: {output_filepath}")
        try:
            with open(output_filepath, 'r', encoding='utf-8') as f_out_check:
                for line_num, line in enumerate(f_out_check, 1):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if 'item_name' in data:
                                processed_item_names.add(data['item_name'])
                            else:
                                print(f"Warning: 'item_name' missing in line {line_num} of {output_filepath}")
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse JSON in line {line_num} of {output_filepath}: '{line[:100]}...'")
            print(f"Found {len(processed_item_names)} item names in existing output file to skip if re-encountered.")
        except Exception as e:
            print(f"Error reading existing output file {output_filepath}: {e}. Will proceed without skipping, potentially creating duplicates if input is reprocessed.")
            processed_item_names = set() # Reset on error to be safe

    try:
        raw_items_generator = load_items_from_jsonl(items_filepath)
        
        # Prepare items_to_process_iterable, considering max_items
        # This list will be iterated by the generator submitting tasks.
        # Filtering for already_processed_names will happen inside generate_processed_items_concurrently.
        all_input_items = list(raw_items_generator) # Materialize all items first
        
        if not all_input_items:
            print("Input file is empty or all lines were skipped during loading.")
            return

        items_to_process_iterable = all_input_items
        if max_items is not None and max_items > 0:
            items_to_process_iterable = all_input_items[:max_items]
            print(f"Considering a maximum of {len(items_to_process_iterable)} items from input (before checking against already processed).")
            if not items_to_process_iterable:
                 print("No items selected after applying max_items limit.")
                 return
        else:
            print(f"Considering all {len(items_to_process_iterable)} loaded items from input (before checking against already processed).")

        processed_items_generator = generate_processed_items_concurrently(
            items_to_process_iterable,
            PROMPT_TEMPLATE,
            max_workers=num_workers,
            already_processed_names=processed_item_names # Pass the set here
        )

        items_saved_count_this_run = save_results_to_jsonl(processed_items_generator, output_filepath)

        # This final check is less critical now as individual functions provide feedback
        # if items_saved_count_this_run == 0:
        #    print("No new items were processed and saved in this run. Check logs for details.")

    except FileNotFoundError:
        print(f"Process aborted: Input file '{items_filepath}' not found.")
    except Exception as e:
        print(f"An unexpected critical error occurred in main execution: {e}")
    finally:
        end_time = time.time()
        print(f"Item labeling process finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # For testing, ensure DEFAULT_ITEMS_FILEPATH exists or create a dummy one.
    # To simulate a re-run, you would run this script once, let it create some output in DEFAULT_OUTPUT_FILEPATH,
    # and then run it again. It should skip the items it processed in the first run.
    
    num_workers_to_use = os.cpu_count() or 2 # Use at least 2 for concurrency if cpu_count is 1 or None
    main(max_items=10, num_workers=num_workers_to_use) # Process up to 10 *new* items