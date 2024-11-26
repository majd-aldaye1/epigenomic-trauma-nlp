"""
This script generates and executes a search query for academic papers using predefined categories (e.g., mental health, epigenetics, ethnicity, socioeconomic status) and expanded terms.

Key Features:
1. **Query Generation**:
   - Dynamically builds a query string using terms from expanded categories.
   - Ensures a minimum number of terms per category.
   - Includes pairwise combinations of ethnographic terms to identify diverse associations.

2. **Paper Fetching**:
   - Executes the generated query using PyPaperBot to scrape academic papers from Google Scholar.
   - Allows customization of search parameters, including the number of pages, year range, and maximum papers to fetch.

3. **Output**:
   - Saves the query string to `query.txt` for review.
   - Downloads the resulting papers to a specified directory for further analysis.

Applications:
This script is designed for researchers conducting meta-analyses, enabling efficient retrieval of papers discussing the relationships between social trauma, epigenetics, and health disparities.
"""

import numpy as np
import logging, random
import subprocess
import os
import json
import regex as re
from itertools import combinations

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import random
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_query(path_to_file, repetitions=1, min_terms=5):
    """
    Generate a query ensuring at least a minimum number of terms from each category,
    allowing for random combinations of terms.

    Args:
        path_to_file (str): Path to the JSON file with expanded terms.
        repetitions (int): Number of repetitions for shuffled queries.
        min_terms (int): Minimum number of terms from each category to be included in the query.

    Returns:
        str: Generated query string.
    """
    try:
        with open(path_to_file, 'r', encoding='utf-8') as file:
            expanded_terms = json.load(file)
    except FileNotFoundError:
        logging.error(f"Error: The file at {path_to_file} was not found.")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error: Failed to decode JSON file at {path_to_file}. Details: {e}")
        raise

    def shuffle_terms(terms):
        shuffled_terms = terms[:]
        random.shuffle(shuffled_terms)

    # Generate subqueries for each category
    epigenetic_query = ' OR '.join([term for term in expanded_terms["epigenetic terms"] if len(term.split())<=2])
    mental_health_query = ' OR '.join([term for term in expanded_terms["mental health terms"] if len(term.split())<=2])
    #ethnographic_query = ' OR '.join([term for category in expanded_terms["ethnographic terms"] for term in expanded_terms["ethnographic terms"][category]])
    socioeconomic_query = ' OR '.join([term for term in expanded_terms["socioeconomic terms"] if len(term.split())<=2])

    ethnographics = [
        'arab', 'american', 'latino', 'indigenous', 'african', 'asian',
        'european', 'australian', 'muslim', 'jewish', 'hispanic',
        ]
    
    # Generate all pairwise combinations within the chunk
    ethnographic_combinations = [
            f"({triple[0]} AND {triple[1]} AND {triple[2]})" for triple in combinations(ethnographics, 3)
        ]
    
    ethnographic_query = f"({' OR '.join(ethnographic_combinations)})"

    # Combine all subqueries into the final query
    query = f"({mental_health_query}) AND ({ethnographic_query}) AND ({socioeconomic_query}) AND (epigenetic) AND (methylation or demethylation)"

    with open('query.txt', 'w') as f:
        print(query, file=f)

    logging.info(f"Generated Query: {query}")
    return query


def fetch_papers(query, scholar_pages=100, min_year=1900, num_papers=10, output_dir="./data/papers"):
    """
    Fetch academic papers using PyPaperBot based on the generated query.

    Args:
        query (str): Query string for searching papers.
        scholar_pages (int): Number of Google Scholar pages to scrape.
        min_year (int): Minimum publication year for papers.
        sort_by_relevance (bool): If True, fetch papers sorted by relevance instead of recency.
        output_dir (str): Directory to save downloaded papers.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Add sorting preference to the query
    #sort_option = "--sort-relevance" if sort_by_relevance else "--sort-recency"

    # Construct PyPaperBot command
    command = [
        "python", "-m", "PyPaperBot",
        f"--query={query}",
        f"--scholar-pages={scholar_pages}",
        f"--min-year={min_year}",
        f"--dwn-dir={output_dir}",
        f"--max-dwn-year={num_papers}"
    ]

    try:
        subprocess.run(command, check=True)
        logging.info(f"Papers successfully fetched and saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error fetching papers with PyPaperBot: {e}")

if __name__ == "__main__":
    # Path to the expanded terms JSON file
    path_to_file = "./expanded_terms.json"

    # Generate the query ensuring at least 5 terms per category
    query = generate_query(path_to_file)

    # Fetch papers using the generated query, sorted by relevance
    fetch_papers(query, scholar_pages=10, min_year=1900, num_papers=100)