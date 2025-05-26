import json
from itertools import islice
from bns.utils.openai_client_azure import prompt_model # Your actual import

# --- Constants ---
PROMPT_TEMPLATE = """Product name: Colgate Total Whitening Toothpaste 120g
Result: {{"brand_name": "Colgate", "item_type": "Toothpaste"}}

Product name: Extra Virgin Olive Oil 500ml
Result: {{"brand_name": null, "item_type": "Olive Oil"}}

Product name: 0308069710162006000999
Result: {{"brand_name": null, "item_type": null}}

Return strictly the JSON result for the below product name.

Product name: {name}
"""
DEFAULT_ITEMS_FILEPATH = 'bns/proj2/data/items.json'
DEFAULT_OUTPUT_FILEPATH = 'bns/proj2/data/branded_items.jsonl' # Changed extension to .jsonl

# --- Data Loading ---
def load_items_from_jsonl(filepath: str):
    """
    Loads items from a JSONL file, yielding item name and count.
    Each line in the file is expected to be a JSON object.
    """
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
                        print(f"Warning: Skipping line {line_number} in '{filepath}' due to missing 'item_name' or 'item_ct': {line}")
                        continue
                    yield (item_name, item['item_ct'])
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_number} in '{filepath}' due to JSON decoding error: {e}. Line content: '{line}'")
                except KeyError as e:
                    print(f"Warning: Skipping line {line_number} in '{filepath}' due to missing key {e}. Line content: '{line}'")
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        raise # Re-raise as this is critical for the script to run

# --- Prompt Engineering ---
def format_llm_prompt(item_name: str, template: str = PROMPT_TEMPLATE) -> str:
    """Formats the prompt for the LLM using the provided item name and template."""
    return template.format(name=item_name)

# --- LLM Interaction and Response Parsing ---
def get_item_labels_from_llm(item_name: str, prompt_template: str = PROMPT_TEMPLATE):
    """
    Gets brand name and item type from the LLM for a given item name.
    Handles LLM call and JSON parsing of the response.
    """
    prompt = format_llm_prompt(item_name, prompt_template)
    try:
        raw_response = prompt_model(prompt)
    except Exception as e:
        print(f"Error during LLM call for item '{item_name}': {e}")
        return None, None

    try:
        label_data = json.loads(raw_response)
        brand_name = label_data.get("brand_name")
        item_type = label_data.get("item_type")
        return brand_name, item_type
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM JSON response for item '{item_name}': {e}. Response: '{raw_response}'")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred while parsing LLM response for item '{item_name}': {e}")
        return None, None

# --- Data Processing (Generator) ---
def generate_processed_items(items_iterable, prompt_template: str = PROMPT_TEMPLATE):
    """
    Processes an iterable of items, gets labels from LLM, and yields structured results.
    """
    for name, ct in items_iterable:
        brand_name, item_type = get_item_labels_from_llm(name, prompt_template)
        processed_item = {
            "item_name": name,
            "brand_name": brand_name,
            "item_type": item_type,
            "item_ct": ct
        }
        print(f"Processed: Name='{name}', Brand='{brand_name}', Type='{item_type}', Count={ct}")
        yield processed_item

# --- Data Saving (JSONL) ---
def save_results_to_jsonl(processed_items_generator, output_path: str) -> int:
    """
    Saves items from a generator to an output file in JSONL format.
    Each item is written as a new line.
    Returns the count of items saved.
    """
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
    return count_saved # Return count even if it's 0 or partial due to error before file opening

# --- Main Execution ---
def main(items_filepath: str = DEFAULT_ITEMS_FILEPATH, output_filepath: str = DEFAULT_OUTPUT_FILEPATH, max_items: int = None):
    """
    Main function to load, process, and save item labels in JSONL format.
    """
    print(f"Starting item labeling process...")
    print(f"Input file: {items_filepath}")
    print(f"Output file: {output_filepath}")

    try:
        # 1. Load items (generator)
        raw_items_generator = load_items_from_jsonl(items_filepath)

        # 2. Optionally limit the number of items to process (e.g., for testing)
        if max_items is not None and max_items > 0 :
            items_to_process = islice(raw_items_generator, max_items)
            print(f"Processing a maximum of {max_items} items.")
        else:
            items_to_process = raw_items_generator
            print("Processing all available items from the input file.")

        # 3. Process items (generator)
        # Renamed process_items to generate_processed_items to emphasize it's a generator
        processed_items_generator = generate_processed_items(items_to_process, PROMPT_TEMPLATE)

        # 4. Save results to JSONL
        items_saved_count = save_results_to_jsonl(processed_items_generator, output_filepath)

        if items_saved_count == 0 and (max_items is None or max_items > 0):
             # Check if input items were actually found if max_items wasn't 0
            with open(items_filepath, 'r', encoding='utf-8') as f_check:
                if not any(line.strip() for line in f_check):
                    print("Warning: Input file was empty or contained only whitespace.")
                else:
                    print("No items were successfully processed and saved. Check warnings above.")

    except FileNotFoundError:
        # load_items_from_jsonl already prints, this is if it's re-raised to here
        print(f"Process aborted: Input file '{items_filepath}' not found.")
    except Exception as e:
        print(f"An unexpected critical error occurred in main execution: {e}")

    print("Item labeling process finished.")

if __name__ == "__main__":
    main(max_items=100)