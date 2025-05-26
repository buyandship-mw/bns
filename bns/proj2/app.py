import json
from itertools import islice
from bns.utils.openai_client_azure import prompt_model

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
DEFAULT_ITEMS_FILEPATH = 'bns/proj2/data/items.json' # Or your actual path
DEFAULT_OUTPUT_FILEPATH = 'bns/proj2/data/branded_items.json' # Or your actual path

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
                        print(f"Warning: Skipping line {line_number} due to missing 'item_name' or 'item_ct': {line}")
                        continue
                    yield (item_name, item['item_ct'])
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_number} due to JSON decoding error: {e}. Line content: '{line}'")
                except KeyError as e:
                    print(f"Warning: Skipping line {line_number} due to missing key {e}. Line content: '{line}'")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise

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
        raw_response = prompt_model(prompt) # Actual call to the LLM
    except Exception as e: # Catch potential errors from prompt_model itself
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
    except Exception as e: # Catch any other unexpected errors during parsing
        print(f"An unexpected error occurred while parsing LLM response for item '{item_name}': {e}")
        return None, None

# --- Data Processing ---
def process_items(items_iterable, prompt_template: str = PROMPT_TEMPLATE):
    """
    Processes a list of items, gets labels from LLM, and yields structured results.
    """
    results = []
    for name, ct in items_iterable:
        brand_name, item_type = get_item_labels_from_llm(name, prompt_template)
        results.append({
            "item_name": name, # Good to keep original name for reference
            "brand_name": brand_name,
            "item_type": item_type,
            "item_ct": ct
        })
        print(f"Processed item: {name}, Brand: {brand_name}, Type: {item_type}, Count: {ct}")
    return results

# --- Data Saving ---
def save_results_to_json(results: list, output_path: str):
    """Saves the list of results to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved {len(results)} labeled items to {output_path}")
    except IOError as e:
        print(f"Error saving results to {output_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving results: {e}")

# --- Main Execution ---
def main(items_filepath: str, output_filepath: str, max_items: int = None):
    """
    Main function to load, process, and save item labels.
    """
    print("Starting item labeling process...")

    # 1. Load items
    raw_items_generator = load_items_from_jsonl(items_filepath)

    # 2. Optionally limit the number of items to process (e.g., for testing)
    if max_items is not None:
        items_to_process = islice(raw_items_generator, max_items)
        print(f"Processing a maximum of {max_items} items.")
    else:
        items_to_process = raw_items_generator
        print("Processing all items from the input file.")

    # 3. Process items
    labeled_results = process_items(items_to_process, PROMPT_TEMPLATE)

    # 4. Save results
    if labeled_results: # Only save if there are results
        save_results_to_json(labeled_results, output_filepath)
    else:
        print("No items were processed or no results generated.")

    print("Item labeling process finished.")

if __name__ == "__main__":
    main(items_filepath=DEFAULT_ITEMS_FILEPATH,
         output_filepath=DEFAULT_OUTPUT_FILEPATH,
         max_items=10) # Use a number like 10 for testing, or None for full run