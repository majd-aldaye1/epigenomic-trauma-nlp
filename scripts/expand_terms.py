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

import wikipediaapi
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import logging
import json


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
logging.info("SentenceTransformer model loaded successfully.")

# Initialize Wikipedia API with a proper user-agent
wiki_wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="majd (ma798@cornell.edu)"
)


def is_valid_title(title):
    """
    Filter Wikipedia titles to exclude irrelevant or non-article entries.
    - Returns False for titles starting with "Help:", "Category:", etc.
    - Ensures titles contain only alphanumeric characters and spaces.

    Args:
        title (str): The title of a Wikipedia page.

    Returns:
        bool: Whether the title is valid for inclusion in the corpus.
    """
    if re.match(r"^(Help:|Category:|File:|Portal:|Template:|Wikipedia:)", title):
        return False
    if not re.match(r"^[A-Za-z0-9\s\-]+$", title):
        return False
    return True


def fetch_wikipedia_corpus(input_terms, max_depth=1, max_pages=200):
    """
    Dynamically fetch a set of related terms from Wikipedia based on input terms.
    - Traverses links from the initial pages up to `max_depth`.
    - Limits the number of pages fetched to `max_pages`.

    Args:
        input_terms (list of str): List of terms to search on Wikipedia.
        max_depth (int): Depth of link traversal.
        max_pages (int): Maximum number of pages to fetch.

    Returns:
        list: A list of valid Wikipedia page titles.
    """
    corpus = set() # set to ensure unique titles

    def fetch_links(page, depth):
        if depth > max_depth or len(corpus) >= max_pages:
            return
        for title in page.links.keys():
            if is_valid_title(title) and title not in corpus:
                corpus.add(title)
                next_page = wiki_wiki.page(title)
                if next_page.exists():
                    fetch_links(next_page, depth + 1)

    for term in input_terms:
        page = wiki_wiki.page(term)
        if page.exists():
            if is_valid_title(page.title):
                corpus.add(page.title)
            fetch_links(page, depth=1)  # Start at depth 1

    logging.info(f"Fetched {len(corpus)} pages from Wikipedia for terms: {input_terms}")
    return list(corpus)


def generate_similar_terms(term_list, model, corpus, topn=50, per_term=False):
    """
    Generate expanded terms by computing semantic similarity between input terms and a corpus.
    Supports both aggregated similarity and per-term similarity.

    Args:
        term_list (list of str): Initial list of input terms.
        model (SentenceTransformer): Pre-trained SentenceTransformer model.
        corpus (list of str): List of potential terms to expand into.
        topn (int): Number of top similar terms to return.
        per_term (bool): If True, compute top similar terms for each individual term in the list.

    Returns:
        dict or list: If per_term is True, returns a dictionary of top terms per input term.
                      Otherwise, returns a single list of top `topn` similar terms.
    """
    logging.info(f"Generating terms for input list: {term_list}")

    # Encode the corpus terms and validate
    corpus_embeddings = []
    valid_corpus = []
    for term in corpus:
        try:
            term_embedding = model.encode(term)
            corpus_embeddings.append(term_embedding)
            valid_corpus.append(term)
        except Exception as e:
            logging.warning(f"Skipping term '{term}': {e}")

    if not corpus_embeddings:
        logging.warning("Empty or invalid corpus embeddings. Returning empty results.")
        return {term: [] for term in term_list} if per_term else []

    logging.info(f"Valid corpus size: {len(valid_corpus)}")
    corpus_embeddings = np.array(corpus_embeddings)  # Ensure embeddings are NumPy arrays

    # Handle per-term similarity (individual processing)
    if per_term:
        top_similar_terms = {}

        for input_term in term_list:
            try:
                input_embedding = model.encode(input_term)
                similarities = util.cos_sim(input_embedding, corpus_embeddings).cpu().numpy()[0]
                sorted_indices = np.argsort(similarities)[::-1]
                top_terms = [valid_corpus[i].lower() for i in sorted_indices[:topn]]

                # Ensure core term is included
                if input_term not in top_terms:
                    top_terms = [input_term.lower()] + top_terms[:topn-1]  
                
                top_similar_terms[input_term] = top_terms
                logging.info(f"Top terms for '{input_term}': {top_terms}")

            except Exception as e:
                logging.error(f"Error processing term '{input_term}': {e}")
                top_similar_terms[input_term] = [input_term]  
        return top_similar_terms

    # Handle aggregated similarity (all input terms combined)
    else:
        # Compute combined embedding for the input list
        term_embeddings = [model.encode(term) for term in term_list]
        if not term_embeddings:
            raise ValueError("No embeddings could be computed for the input list.")
        list_embedding = np.mean(term_embeddings, axis=0)  # Aggregate embeddings

        # Compute cosine similarities
        try:
            similarities = util.cos_sim(list_embedding, corpus_embeddings).cpu().numpy()[0]
            logging.info(f"First few similarities: {similarities[:10]}")
        except Exception as e:
            raise ValueError(f"Error computing similarities: {e}")

        if similarities is None or len(similarities) == 0:
            raise ValueError("No similarities could be computed.")

        # Sort terms by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        top_similar_terms = [valid_corpus[i].lower() for i in sorted_indices[:topn]]

        logging.info(f"Top {topn} similar terms: {top_similar_terms}")
        return top_similar_terms


def expand_terms_for_query(terms, max_depth=1, topn=50, per_term = False):
    """
    Expand input terms using Wikipedia and semantic similarity.

    Args:
        terms (list of str): List of terms to expand.
        max_depth (int): Depth of Wikipedia traversal.
        topn (int): Number of top similar terms to return.

    Returns:
        list: Expanded list of terms.
    """
    logging.info(f"Expanding terms for: {terms}")
    corpus = fetch_wikipedia_corpus(terms, max_depth=max_depth, max_pages=200)
    if not corpus:
        logging.warning(f"The fetched corpus is empty for terms: {terms}.")
        return {term: [] for term in terms}  # Return an empty dictionary for each term

    # Generate top similar terms for each input term
    return generate_similar_terms(terms, model, corpus, topn, per_term=per_term)


def expand_and_save_to_json(mental_health_terms, epigenetic_terms, ethnographic_terms, socioeconomic_terms):
    """
    Expand terms for predefined categories and save them to a JSON file, ensuring all terms are lowercase.

    Args:
        mental_health_terms (list of str): Terms related to mental health.
        epigenetic_terms (list of str): Terms related to epigenetics.
        ethnographic_terms (dict of list): Terms related to race and ethnicity.
        socioeconomic_terms (list of str): Terms related to socioeconomic factors.
    """
    expanded_terms = {}

    # Expand terms and ensure lowercase
    expanded_terms["Mental Health Terms"] = [
        term.lower() for term in expand_terms_for_query(mental_health_terms)
    ]
    expanded_terms["Epigenetic Terms"] = [
        term.lower() for term in expand_terms_for_query(epigenetic_terms)
    ]
    expanded_terms["Socioeconomic Terms"] = [
        term.lower() for term in expand_terms_for_query(socioeconomic_terms)
    ]

    # Maintain the nested structure for ethnographic terms and normalize to lowercase
    expanded_terms["Ethnographic Terms"] = {
        category.lower(): [
            term.lower() for term in expand_terms_for_query(terms, topn=10, per_term=True)
        ]
        for category, terms in ethnographic_terms.items()
    }

    # Save expanded terms to a file
    with open("expanded_terms.json", "w", encoding="utf-8") as outf:
        json.dump(expanded_terms, outf, indent=4)
    logging.info("Expanded terms saved to './scripts/expanded_terms.json'.")



if __name__ == "__main__":
    # Core terms for querying
    mental_health_terms = [
        "depression", "bipolar", "PTSD", "anxiety", 
        "suicide", "generational trauma", "chronic stress"
    ]
    epigenetic_terms = [
        "methylation", "demethylation", "CpG islands", "5mC", 
        "histone modification", "H3K27me3", "epigenetic", "epigenomics",
        "BDNF", "SLC6A4", "FKBP5", "OXTR", "stress response", 
        "HPA axis dysregulation", "childhood abuse"
    ]
    ethnographic_terms = {
        "African descent": [
            "African person", "African-American person", "Black person", "Black diaspora", "African-American mother"
        ],
        "Latino/Hispanic descent": [
            "Latino person", "Hispanic individual", "Latino community", "Hispanic person"
        ],
        "Asian descent": [
            "Asian person", "Asian-American person", "East Asian person", "South Asian person"
        ],
        "Indigenous descent": [
            "Indigenous person", "Native American person", "First Nations person", "Indigenous peoples"
        ],
        "Arab descent": [
            "Arab person", "Middle-Eastern individual", "Muslim person", "Arab-American person"
        ],
        "European descent": [
            "European person", "Caucasian person", "White person", "White American", "Anglo-American"
        ]
    }
    socioeconomic_terms = [
    "low-income", "middle-income", "high-income", "below poverty", "above poverty"
    ]

    # Generate the query
    expand_and_save_to_json(mental_health_terms, epigenetic_terms, ethnographic_terms, socioeconomic_terms)
