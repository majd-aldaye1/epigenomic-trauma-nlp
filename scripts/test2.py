import wikipediaapi
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

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

def fetch_wikipedia_corpus(input_terms, max_depth=2, max_pages=200):
    """
    Dynamically fetch a corpus of related terms from Wikipedia up to a given depth.

    Args:
        input_terms (list): Input terms to start the search.
        max_depth (int): Maximum depth to explore links.
        max_pages (int): Maximum number of pages to fetch.

    Returns:
        list: A list of terms from Wikipedia.
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

    Args:
        term_list (list): List of input terms.
        model (SentenceTransformer): SentenceTransformer model.
        corpus (list): Combined corpus of terms (Wikipedia + MeSH).
        topn (int): Number of similar terms to return.

    Returns:
        list: Expanded terms.
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
        logging.info(f"Type of similarities: {type(similarities)}")
        logging.info(f"Shape of similarities: {similarities.shape}")
        logging.info(f"First few similarities: {similarities[:10]}")
    except Exception as e:
        raise ValueError(f"Error computing similarities: {e}")

    if similarities is None or len(similarities) == 0:
        raise ValueError("No similarities could be computed.")

    # Ensure similarities is a 1D array
    if not isinstance(similarities, np.ndarray) or len(similarities.shape) != 1:
        raise ValueError(f"Similarities is malformed: {similarities}")

    # Sort terms by similarity
    sorted_indices = np.argsort(similarities)[::-1]
    top_similar_terms = [valid_corpus[i] for i in sorted_indices[:topn]]

    return top_similar_terms

if __name__ == "__main__":
    # Input terms
    term_lists = {
        "Mental Health Terms": ["depression", "bipolar", "PTSD", "anxiety", "suicide"],
        "Epigenetic Terms": ["DNA methylation", "histone modification", "gene expression"],
        "Ethnographic Terms": [
            "race", "ethnicity", "African American", "Latino", "Caucasian",
            "Asian", "Native American", "Hispanic", "Indigenous", "Arab", "Middle Eastern"
        ],
        "Socioeconomic Terms": [
            "socioeconomic status", "income inequality", "poverty", "social class",
            "education disparity", "economic hardship"
        ],
    }

    # Open the output file
    with open("expanded_terms_all_lists.txt", "w", encoding="utf-8") as outf:
        for list_name, terms in term_lists.items():
            # Fetch corpus for each list
            logging.info(f"Fetching corpus for {list_name}...")
            corpus = fetch_wikipedia_corpus(terms)
            if not corpus:
                logging.warning(f"The fetched corpus for {list_name} is empty. Skipping.")
                continue

            # Generate expanded terms
            try:
                expanded_terms = generate_similar_terms(terms, model, corpus)
                outf.write(f"{list_name}:\n")
                for i in range(0, len(expanded_terms), 3):
                    outf.write(f"{expanded_terms[i:i+3]}\n")
                outf.write("\n")
                logging.info(f"Expanded terms for {list_name} written to file.")
            except ValueError as e:
                logging.error(f"Error generating similar terms for {list_name}: {e}")
