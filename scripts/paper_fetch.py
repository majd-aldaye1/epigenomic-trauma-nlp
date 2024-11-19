import wikipediaapi
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import logging
import subprocess
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
logging.info("Model loaded successfully.")

# Initialize Wikipedia API with a proper user-agent
wiki_wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="majd (ma798@cornell.edu)"
)

def is_valid_title(title):
    """
    Filter Wikipedia titles (exclude irrelevant entries).
    """
    if re.match(r"^(Help:|Category:|File:|Portal:|Template:|Wikipedia:)", title):
        return False
    if not re.match(r"^[A-Za-z0-9\s\-]+$", title):
        return False
    return True

def fetch_wikipedia_corpus(input_terms, max_depth=1, max_pages=200):
    """
    Dynamically fetch a corpus of related terms from Wikipedia up to a given depth.
    """
    corpus = set()

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

    return list(corpus)

def generate_similar_terms(term_list, model, corpus, topn=50):
    """
    Generate expanded terms using a combined corpus.
    """
    logging.info(f"Generating terms for input list: {term_list}")

    # Compute the combined embedding for the input list
    term_embeddings = [model.encode(term) for term in term_list]
    if not term_embeddings:
        raise ValueError("No embeddings could be computed for the input list.")
    list_embedding = np.mean(term_embeddings, axis=0)

    # Encode corpus terms and validate
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
        raise ValueError("No valid embeddings could be computed for the corpus.")

    logging.info(f"Valid corpus size: {len(valid_corpus)}")

    # Convert embeddings to NumPy arrays to avoid tensor warnings
    corpus_embeddings = np.array(corpus_embeddings)
    list_embedding = np.array(list_embedding)

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
    top_similar_terms = [valid_corpus[i] for i in sorted_indices[:topn]]

    return top_similar_terms

def expand_terms_for_query(terms, max_depth=1, topn=50):
    """
    Fetch a Wikipedia corpus and generate expanded terms for querying.
    """
    logging.info(f"Expanding terms for: {terms}")
    corpus = fetch_wikipedia_corpus(terms, max_depth=max_depth, max_pages=200)
    if not corpus:
        logging.warning(f"The fetched corpus is empty for terms: {terms}.")
        return terms  # Fall back to the original terms
    return generate_similar_terms(terms, model, corpus, topn)

def generate_query(mental_health_terms, epigenetic_terms, ethnographic_terms, socioeconomic_terms):
    """
    Generate a query for PyPaperBot using expanded terms.
    """
    expanded_terms = {}

    expanded_terms["Mental Health Terms"] = expand_terms_for_query(mental_health_terms)
    expanded_terms["Epigenetic Terms"] = expand_terms_for_query(epigenetic_terms)
    expanded_terms["Ethnographic Terms"] = expand_terms_for_query(ethnographic_terms)
    expanded_terms["Socioeconomic Terms"] = expand_terms_for_query(socioeconomic_terms)

    # Save expanded terms to a file
    with open("expanded_terms.txt", "w", encoding="utf-8") as outf:
        for category, terms in expanded_terms.items():
            outf.write(f"{category}:\n")
            outf.write(", ".join(terms) + "\n\n")
    logging.info("Expanded terms saved to 'expanded_terms.txt'.")

    # Build a dynamic query using expanded term lists
    query = (
        f"({' OR '.join(expanded_terms['Epigenetic Terms'])}) AND "
        f"({' OR '.join(expanded_terms['Mental Health Terms'])}) OR "
        f"({' OR '.join(expanded_terms['Ethnographic Terms'])}) OR "
        f"({' OR '.join(expanded_terms['Socioeconomic Terms'])})"
    )
    logging.info(f"Generated Query: {query}")
    return query

def fetch_papers(query, scholar_pages=1, min_year=2000, output_dir="./data/testing_folder"):
    """
    Fetch papers using PyPaperBot with the given query.
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
    mental_health_terms = ["depression", "bipolar", "PTSD", "anxiety", "suicide"]
    epigenetic_terms = ["DNA methylation", "histone modification", "gene expression"]
    ethnographic_terms = [
        "race", "ethnicity", "African American", "Latino", "Caucasian",
        "Asian", "Native American", "Hispanic", "Indigenous", "Arab", "Middle Eastern"
    ]
    socioeconomic_terms = [
        "socioeconomic status", "income inequality", "poverty", "social class",
        "education disparity", "economic hardship"
    ]

    # Generate the query
    query = generate_query(mental_health_terms, epigenetic_terms, ethnographic_terms, socioeconomic_terms)

    # Fetch papers using the generated query
    fetch_papers(query)
