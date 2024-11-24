"""
fetch.py 

Purpose:
- Fetches and expands terms related to predefined categories (e.g., Mental Health, Epigenetic, etc.) 
  using semantic similarity.
- Dynamically generates a list of top similar terms for each core term in the categories, 
  enabling enhanced text analysis for the next step in preprocessing.py.

Key Steps:
1. Load core terms for categories like Mental Health, Epigenetic, Ethnographic, and Socioeconomic terms.
2. Use a SentenceTransformer model to calculate semantic similarity between core terms and a corpus 
   (e.g., Wikipedia, other text datasets).
3. Generate a list of top similar terms for each core term.
4. Save the expanded terms for each category in a structured JSON file.

Inputs:
- Core term lists for categories (e.g., ["depression", "anxiety", "PTSD"] for Mental Health).
- Corpus of potential related terms (e.g., scraped Wikipedia links or other text data).

Outputs:
- JSON file (`top_similar_terms.json`) containing:
  {
      "Mental Health Terms": {"depression": ["stress", "psychosis", ...], ...},
      "Epigenetic Terms": {"methylation": ["CpG islands", "DNA modifications", ...], ...},
      ...
  }

How to Use:
- This script generates a JSON file of expanded terms, which is consumed by `preprocessing.py` for term matching.
"""


import numpy as np
import logging
import subprocess
import os
import json


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_query(path_to_file):
    try:
        with open(path_to_file, 'r', encoding='utf-8') as file:
            expanded_terms = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file at {path_to_file} was not found.")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON file at {path_to_file}. Details: {e}")
        raise
    
    # Flatten nested Ethnographic Terms into a single list of terms
    ethnographic_flattened = [
        term for category_terms in expanded_terms["Ethnographic Terms"].values()
        for term in category_terms
    ]

    # Build a dynamic query using following logic:
    # 1. Papers must contain at least one Epigenetic Term.
    # 2. Papers must contain at least one Mental Health Term.
    # 3. Must also contain at least one term from either Ethnographic OR Socioeconomic Terms.
    query = (
        f"({' OR '.join(expanded_terms['Epigenetic Terms'])}) AND "  # Epigenetic terms (required)
        f"({' OR '.join(expanded_terms['Mental Health Terms'])}) AND "  # Mental health terms (required)
        f"({' OR '.join(ethnographic_flattened)} OR "  # Flattened ethnographic terms...
        f"{' OR '.join(expanded_terms['Socioeconomic Terms'])})"  # ...or socioeconomic terms
    )

    logging.info(f"Generated Query: {query}")
    return query


def fetch_papers(query, scholar_pages=10, min_year=2000, output_dir="./data/papers"):
    """
    Fetch academic papers using PyPaperBot based on the generated query.

    Args:
        query (str): Query string for searching papers.
        scholar_pages (int): Number of Google Scholar pages to scrape.
        min_year (int): Minimum publication year for papers.
        output_dir (str): Directory to save downloaded papers.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Construct PyPaperBot command
    command = [
        "python", "-m", "PyPaperBot",
        f"--query={query}",
        f"--scholar-pages={scholar_pages}",
        f"--min-year={min_year}",
        f"--dwn-dir={output_dir}"
    ]

    try:
        subprocess.run(command, check=True)
        logging.info(f"Papers successfully fetched and saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error fetching papers with PyPaperBot: {e}")


if __name__ == "__main__":
    # Core terms for querying
    path_to_file = "./expanded_terms.json"

    # Generate the query
    query = generate_query(path_to_file)

    # Fetch papers using the generated query
    fetch_papers(query)
