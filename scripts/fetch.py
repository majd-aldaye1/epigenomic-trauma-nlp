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

How to Use:
1. Update `path_to_file` with the path to the expanded terms JSON file.
2. Run the script to generate a query and fetch papers.
3. The papers will be saved in the specified output directory (`data/papers` by default).
"""
import logging, random, os, json, subprocess, shlex
from itertools import combinations

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

    def shuffle_terms(terms, num_terms):
        """Shuffle and return a specified number of terms."""
        shuffled_terms = terms[:]
        random.shuffle(shuffled_terms)
        return shuffled_terms[:num_terms]

    # Generate subqueries for each category
    epigenetic_query = ' OR '.join(shuffle_terms(expanded_terms["epigenetic terms"], min_terms))
    mental_health_query = ' OR '.join(shuffle_terms(expanded_terms["mental health terms"], 2))
    socioeconomic_query = ' OR '.join(shuffle_terms(expanded_terms["socioeconomic terms"], 1))

    ethnographics = [
        'arab', 'american', 'latino', 'indigenous', 'african', 'asian',
        'european', 'australian', 'hispanic',
        ]
    # Generate all pairwise combinations within the chunk, limit up to 5 combinations
    ethnographic_combinations = [f"({pair[0]} OR {pair[1]})" for pair in combinations(ethnographics, 2)][:3]  # Limit to 5 pairs    
    ethnographic_query = f"({' OR '.join(ethnographic_combinations)})"

    # Combine all subqueries into the final query
    query = f"({mental_health_query}) AND ({ethnographic_query}) AND ({socioeconomic_query}) AND ({epigenetic_query}) AND (epigenetic) AND (methylation or demethylation)"

    with open('query.txt', 'w') as f:
        f.write(query)
    logging.info("Query saved to 'query.txt'.")
    logging.info(f"Generated Query: {query}")
    return query


# Ensure that special characters in the query are properly escaped
def escape_query(query):
    """Escape special characters in the query."""
    return shlex.quote(query)


def fetch_papers(query, scholar_pages=500, min_year=1900, num_papers=10, output_dir="./data/papers"):
    """
    Fetch academic papers using PyPaperBot based on the generated query.

    Args:
        query (str): Query string for searching papers.
        scholar_pages (int): Number of Google Scholar pages to scrape.
        min_year (int): Minimum publication year for papers.
        sort_by_relevance (bool): If True, fetch papers sorted by relevance instead of recency.
        output_dir (str): Directory to save downloaded papers.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Add sorting preference to the query
    #sort_option = "--sort-relevance" if sort_by_relevance else "--sort-recency"
    
    # Escape the query string
    query = escape_query(query)

    # Construct PyPaperBot command
    command = [
        "python", "-m", "PyPaperBot",
        f'--query="{query}"',  # Use double quotes for the query
        f"--scholar-pages={scholar_pages}",
        f"--min-year={min_year}",
        f"--dwn-dir={output_dir}"
    ]
    logging.info(f"Executing PyPaperBot with command: {' '.join(command)}")

    # Execute the command and handle errors
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"Subprocess Output: {result.stdout}")
        logging.info(f"Papers successfully fetched and saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess Error Output: {e.stderr}")
        logging.error(f"Error fetching papers with PyPaperBot: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error occurred while fetching papers: {e}")
        raise


if __name__ == "__main__":
    # Path to the expanded terms JSON file
    path_to_file = "./scripts/json/expanded_terms.json"

    # Step 1: Generate the query ensuring at least 5 terms per category
    query = generate_query(path_to_file)

    # Step 2: Fetch papers using the generated query, sorted by relevance
    fetch_papers(query, scholar_pages=200, min_year=2000)