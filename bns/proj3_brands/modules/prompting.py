# prompting.py

def build_prompt(input_text: str) -> str:
    """
    Generate a prompt for the model based on the item name or URL.
    """
    return (
        f'Extract the brand name: "{input_text}"\n\n'
        'Respond in JSON: {"name": <value>} or {"error": <reason>}'
    )
