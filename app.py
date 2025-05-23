import csv

from openai_client_azure import prompt_model

def get_prompt(company):
    return f"""
Collect a list of 2-4 recent investments made by the selected company.

For each investee, provide:
- Name
- Region (country code)
- Industry (short description)
- Series type (e.g., Seed, A, B, C, Public, etc.)

Format the output as a comma-separated list. For each investee, use this structure:
Investee Name (Region, Industry, Series)

Example output:
\""" 
Ably Corp. (KR, fashion e-commerce, unknown), Vpon (TW, media & information services, Seed), Resilience (US, healthcare, A), ORCA-TECH (CN, AI & robotics, B), Thunes (SG, B2B payments, C), Klook (HK, travel & experiences, E)
\""" 

Company: {company}
"""

def read_companies(filename):
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        companies = [row[0].strip() for row in reader if row]
    if companies and companies[0].lower() in {"company", "company name", "name"}:
        companies = companies[1:]
    return companies

def main():
    input_file = "companies.csv"
    output_file = "investments_output.txt"
    companies = read_companies(input_file)

    with open(output_file, "w", encoding="utf-8") as out_f:
        for company in companies:
            prompt = get_prompt(company)
            try:
                result = prompt_model(prompt)
                out_f.write(f"Company: {company}\n{result.strip()}\n\n")
                print(f"Processed: {company}")
            except Exception as e:
                out_f.write(f"Company: {company}\nERROR: {str(e)}\n\n")
                print(f"Error processing {company}: {e}")

if __name__ == "__main__":
    main()