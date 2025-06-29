import os
import csv
import pandas as pd
import json
import configparser
from openai import OpenAI
import re

# Read API key from config file
config = configparser.ConfigParser()
config.read('config.ini')
api_key = config['openai']['api_key']

# Create a new OpenAI client
client = OpenAI(api_key=api_key)

def prompt_model(prompt):
    """
    Sends the prompt to the model and returns the response.
    """
    completion = client.responses.create(
        model="gpt-4o-mini",
        tools=[{"type": "web_search_preview"}],
        input=prompt
    )
    return completion.output_text

def classify_ticket(name):
    """
    Calls the OpenAI API to estimate ticket size and rationale for a given investor name.
    Expects the model to return a JSON object with 'ticket_size' and 'rationale' fields.
    """
    prompt = (
        "You are a professional investment analyst with access to public investment databases (e.g., Crunchbase, PitchBook, Dealroom) and press releases.\n\n"
        f"Investor: {name}\n\n"
        "For the investor above, perform these steps:\n"
        "1. Find all **direct equity investments made by the investor** (exclude any investments into other funds) closed since 2020.\n"
        "2. For each deal, list:\n"
        "- Company name\n"
        "- **Amount invested by investor** (USD, integer)\n"
        "- Year of the deal\n"
        "- Credible source URL\n"
        "3. From that list, determine the investor’s **single largest investment** (i.e. the highest amount they put into any one company).\n"
        "4. If an exact amount isn’t publicly disclosed, provide a well-reasoned ticket size estimate (based on comparable deals or typical ticket sizes) and cite your rationale.\n"
        "Respond **only** in valid JSON, matching this schema exactly (no extra text):\n"
        "```json\n"
        '{ "investor": <string>, "ticket_size": <integer>, "rationale": "<URL or brief explanation for this figure>" }'
    )

    try:
        raw_output = prompt_model(prompt)
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip(), flags=re.IGNORECASE)

        # Parse JSON
        data = json.loads(cleaned)

        # Extract values
        investor = data.get("investor")
        rationale = data.get("rationale")
        ticket_size = data.get("ticket_size")

        print(f"Investor: {investor}, Ticket size: {ticket_size}, Rationale: {rationale}")
        
        return investor, ticket_size, rationale
    except Exception as e:
        # Fallback or logging
        return None, f"Error: {str(e)}"

# Paths
input_path = 'data/investor_list.csv'
output_path = 'data/output.csv'

# Read input CSV
df = pd.read_csv(input_path)

# Open output CSV in append mode
with open(output_path, 'a', newline='') as csvfile:
    fieldnames = ['Investor', 'Ticket size (USD)', 'Rationale']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write header only if file is empty
    if os.stat(output_path).st_size == 0:
        writer.writeheader()

    # Iterate and process each row
    for _, row in df.iterrows():
        name = row['Potential investor']
        investor, ticket_size, rationale = classify_ticket(name)
        
        # Write directly to CSV after each generation
        writer.writerow({
            'Investor': investor,
            'Ticket size (USD)': ticket_size,
            'Rationale': rationale
        })

print(f"Results are being written to {output_path} as they are generated.")