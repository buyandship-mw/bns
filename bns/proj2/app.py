import json
from itertools import islice
import concurrent.futures
import os
import time

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

# --- Data Loading (remains the same) ---
def load_items_from_jsonl(filepath: str):
    # ... (your existing load_items_from_jsonl function)
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
                except json.JSONDecodeError as e:
                    # print(f"Warning: Skipping line {line_number} in '{filepath}' due to JSON decoding error: {e}. Line content: '{line}'")
                    continue
                except KeyError as e:
                    # print(f"Warning: Skipping line {line_number} in '{filepath}' due to missing key {e}. Line content: '{line}'")
                    continue
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        raise

# --- Prompt Engineering (remains the same) ---
def format_llm_prompt(item_name: str, template: str = PROMPT_TEMPLATE) -> str:
    return template.format(name=item_name)

# --- LLM Interaction and Response Parsing ---
def get_item_labels_from_llm(item_name: str, prompt_template: str = PROMPT_TEMPLATE):
    """
    Gets brand name and item type from the LLM for a given item name.
    Handles LLM call and JSON parsing of the response, including stripping Markdown fences.
    """
    prompt = format_llm_prompt(item_name, prompt_template)
    try:
        raw_response = prompt_model(prompt) # Actual call to the LLM
    except Exception as e: # Catch potential errors from prompt_model itself
        print(f"Error during LLM call for item '{item_name}': {e}")
        return None, None

    # Attempt to clean the raw_response from common Markdown fences
    cleaned_response_str = raw_response.strip()

    if cleaned_response_str.startswith("```json") and cleaned_response_str.endswith("```"):
        # Handles ```json ... ```
        # Slice off "```json" from the start and "```" from the end
        json_str_to_parse = cleaned_response_str[len("```json") : -len("```")].strip()
    elif cleaned_response_str.startswith("```") and cleaned_response_str.endswith("```"):
        # Handles ``` ... ``` (generic code block)
        # Slice off "```" from the start and "```" from the end
        json_str_to_parse = cleaned_response_str[len("```") : -len("```")].strip()
    else:
        # Assume the response is already plain JSON or will fail parsing as before
        json_str_to_parse = cleaned_response_str

    try:
        # Ensure json_str_to_parse is not empty after potential stripping,
        # which could happen if the LLM returned, e.g., "```json\n```"
        if not json_str_to_parse:
            # This specific error message helps diagnose if stripping resulted in an empty string.
            raise json.JSONDecodeError("Extracted JSON string is empty after stripping fences.", cleaned_response_str, 0)

        label_data = json.loads(json_str_to_parse)
        brand_name = label_data.get("brand_name")
        item_type = label_data.get("item_type")
        return brand_name, item_type
    except json.JSONDecodeError as e:
        # Enhanced error message to show what was attempted to be parsed
        print(f"Error parsing LLM JSON response for item '{item_name}': {e}. "
              f"Attempted to parse (first 200 chars): '{json_str_to_parse[:200]}...' "
              f"Raw response (first 200 chars): '{raw_response[:200]}...'")
        return None, None
    except Exception as e: # Catch any other unexpected errors during parsing
        print(f"An unexpected error occurred while parsing LLM response for item '{item_name}': {e}")
        return None, None

# --- Helper function for concurrent processing ---
def process_single_item_task(item_data, prompt_template_str):
    """
    Task function to process a single item: get labels and form the result.
    This function will be executed by each thread in the pool.
    """
    name, ct = item_data  # Unpack item data
    brand_name, item_type = get_item_labels_from_llm(name, prompt_template_str)

    if brand_name is None and item_type is None:
        print(f"Skipping (in thread): Name='{name}' as both brand_name and item_type are null.")
        return None  # Return None for skipped items

    processed_item = {
        "item_name": name,
        "brand_name": brand_name,
        "item_type": item_type,
        "item_ct": ct
    }
    print(f"Processed (in thread): Name='{name}', Brand='{brand_name}', Type='{item_type}', Count={ct}")
    return processed_item

# --- Data Processing (Generator) - Modified for Concurrency ---
def generate_processed_items_concurrently(items_iterable, prompt_template_str: str = PROMPT_TEMPLATE, max_workers: int = None):
    """
    Processes an iterable of items concurrently using a ThreadPoolExecutor,
    gets labels from LLM, and yields structured results.
    Items where both brand_name and item_type are null (None) will be skipped.
    """
    if max_workers is None:
        # Default to number of CPU cores as a starting point for max_workers.
        # For I/O bound tasks, this can often be higher.
        # IMPORTANT: TUNE THIS VALUE based on API rate limits and performance.
        max_workers = os.cpu_count() or 1 # Default to 1 if os.cpu_count() is None
        print(f"Using up to {max_workers} worker threads (default based on CPU cores). Adjust based on API limits.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # The `map` function will apply `process_single_item_task` to each item in `items_iterable`.
        # It also conveniently handles passing the `prompt_template_str` to each call via a lambda or functools.partial.
        # For simplicity, if `process_single_item_task` only needs one varying argument from the iterable,
        # and other arguments are fixed, we can use a helper or ensure `items_iterable` yields tuples/objects
        # that `process_single_item_task` can unpack.
        
        # Here, items_iterable yields (name, ct). process_single_item_task expects (item_data, prompt_template_str)
        # We can use a lambda to adapt this for executor.map, which expects a function of one argument from the iterable.
        
        # To pass the (constant) prompt_template to the worker function with map:
        # We can't directly pass multiple arguments to the mapped function if one comes from the iterable and other is fixed.
        # So, we'll create futures one by one.
        
        futures = []
        for item_data in items_iterable: # item_data is (name, ct)
            futures.append(executor.submit(process_single_item_task, item_data, prompt_template_str))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result is not None: # Filter out items that were marked for skipping
                    yield result
            except Exception as exc:
                # Handle exceptions that occurred within the thread task if not caught inside process_single_item_task
                # This depends on how robust process_single_item_task is.
                # For now, assume process_single_item_task handles its own errors and returns None or valid data.
                print(f'An item generated an exception: {exc}')
                # Optionally, yield an error object or skip


# --- Data Saving (JSONL) - remains the same ---
def save_results_to_jsonl(processed_items_generator, output_path: str) -> int:
    count_saved = 0
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in processed_items_generator:
                try:
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(json_line + '\n')
                    count_saved += 1
                except TypeError as e:
                    print(f"Error serializing item to JSON: {e}. Item: {item}")
        print(f"Successfully saved {count_saved} labeled items to {output_path}")
        return count_saved
    except IOError as e:
        print(f"Error writing to output file {output_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving results to {output_path}: {e}")
    return count_saved

# --- Main Execution - Modified to call the concurrent generator ---
def main(items_filepath: str = DEFAULT_ITEMS_FILEPATH,
         output_filepath: str = DEFAULT_OUTPUT_FILEPATH,
         max_items: int = None,
         num_workers: int = None): # New parameter for number of workers
    """
    Main function to load, process concurrently, and save item labels in JSONL format.
    """
    print(f"Starting item labeling process...")
    print(f"Input file: {items_filepath}")
    print(f"Output file: {output_filepath}")
    if num_workers:
        print(f"Requested number of worker threads: {num_workers}")

    start_time = time.time() # For timing

    try:
        raw_items_generator = load_items_from_jsonl(items_filepath)

        items_to_process_iterable = raw_items_generator
        if max_items is not None and max_items > 0:
            # Materialize the limited items into a list for the executor,
            # as islice on a generator might be tricky with some executor patterns
            # if not consumed carefully. For `submit` loop, it's fine.
            items_to_process_iterable = list(islice(raw_items_generator, max_items))
            print(f"Processing a maximum of {len(items_to_process_iterable)} items.")
            if not items_to_process_iterable:
                 print("No items to process after applying max_items limit or from file.")
                 return # Exit early if no items
        else:
            # If processing all, and items_iterable is a true generator,
            # be mindful if it's very large. For `submit` loop, it's fine.
            # For testing, it might be better to convert to list if the file is small.
            # For production with huge files, ensure the generator consumption is efficient.
            # items_to_process_iterable = list(raw_items_generator) # Use if you need to know total count beforehand or for some executor patterns
            print("Processing all available items from the input file (or until generator is exhausted).")


        # Pass the PROMPT_TEMPLATE string directly
        processed_items_generator = generate_processed_items_concurrently(
            items_to_process_iterable,
            PROMPT_TEMPLATE, # Pass the template string
            max_workers=num_workers # Pass user-defined num_workers or None for default
        )

        items_saved_count = save_results_to_jsonl(processed_items_generator, output_filepath)

        if items_saved_count == 0 and (max_items is None or (isinstance(items_to_process_iterable, list) and len(items_to_process_iterable) > 0) or max_items > 0 ):
            # Check if input items were actually found if max_items wasn't 0
            # This check is a bit more complex now due to potential list conversion
            is_input_empty = True
            try:
                with open(items_filepath, 'r', encoding='utf-8') as f_check:
                    if any(line.strip() for line in f_check):
                        is_input_empty = False
                if is_input_empty:
                     print("Warning: Input file was empty or contained only whitespace.")
                elif isinstance(items_to_process_iterable, list) and not items_to_process_iterable:
                     print("No items were loaded to process (e.g. all lines skipped or max_items=0).")
                else:
                    print("No items were successfully processed and saved. Check warnings above.")
            except FileNotFoundError: # Should be caught earlier but as a safeguard
                pass
    except FileNotFoundError:
        print(f"Process aborted: Input file '{items_filepath}' not found.")
    except Exception as e:
        print(f"An unexpected critical error occurred in main execution: {e}")
    finally:
        end_time = time.time()
        print(f"Item labeling process finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    num_workers_to_use = os.cpu_count()
    main(max_items=50, num_workers=num_workers_to_use) # Process 100 items with specified workers