from rapidfuzz import fuzz
import logging
from scripts.paper_fetch import fetch_wikipedia_corpus, generate_similar_terms

from sentence_transformers import SentenceTransformer

# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def is_fuzzy_match(term, valid_terms, threshold=65):
    """
    Check if the term is a fuzzy match to any term in a list of valid terms.

    Args:
        term (str): The term to check.
        valid_terms (list of str): The list of valid terms to match against.
        threshold (int): Minimum similarity ratio for a match.

    Returns:
        bool: True if a close match is found, False otherwise.
    """
    for valid_term in valid_terms:
        similarity = fuzz.ratio(term.lower(), valid_term.lower())
        print(f"Comparing '{term}' with '{valid_term}': {similarity}% similarity")
        if similarity > threshold:
            return True
    return False

def test_fuzzy_matching(core_terms, topn=10, threshold=65):
    """
    Test fuzzy matching on a list of test terms against ethnographic terms.

    Args:
        ethnographic_terms (list of str): Core ethnographic terms.
        test_corpus (list of str): List of terms to match against.
        threshold (int): Minimum similarity ratio for a match.

    Returns:
        dict: Dictionary of matched terms for each core ethnographic term.
    """
    # Dynamically expand core terms into a corpus using Wikipedia and embeddings
    expanded_corpus = []
    for core_term in core_terms:
        corpus = fetch_wikipedia_corpus([core_term], max_depth=1)
        expanded_terms = generate_similar_terms([core_term], model, corpus, topn=topn, per_term=True)
        expanded_corpus.extend(expanded_terms[core_term])  # Flatten into one list

    logging.info(f"Expanded Corpus: {expanded_corpus}")

    # Perform fuzzy matching between core terms and expanded corpus
    matched_results = {}
    for core_term in core_terms:
        matches = []
        for test_term in expanded_corpus:
            similarity = fuzz.ratio(core_term.lower(), test_term.lower())
            if similarity > threshold:
                matches.append((test_term, similarity))
        matched_results[core_term] = sorted(matches, key=lambda x: x[1], reverse=True)

    return matched_results


if __name__ == "__main__":
    # Example core ethnographic terms
    ethnographic_terms = [
        "African person", "Black individual", "Asian person", 
        "Hispanic person", "Latino person", "Indigenous person", 
        "Native American", "Arab person", "Caucasian person", "White person"
    ]

    # Example test corpus (e.g., Wikipedia titles or other terms)
    test_corpus = [
        "African people", "Black diaspora", "Asian-American individual", 
        "Latinx community", "Indigenous peoples", "Native tribes",
        "Arab-American", "Arab American", "White American", "Caucasian individual", 
        "Hispanic culture", "African-American mother", "Asian culture"
    ]

    # Test fuzzy matching
    threshold = 65  # Adjust this as needed
    results = test_fuzzy_matching(ethnographic_terms, test_corpus, threshold)

    # Print results
    print("\nFuzzy Matching Results:")
    for ethnographic_term, matches in results.items():
        print(f"\nCore Term: {ethnographic_term}")
        for match, similarity in matches:
            print(f"  - Match: {match} (Similarity: {similarity}%)")
