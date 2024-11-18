from nltk.corpus import wordnet as wn
from itertools import chain

def get_similar_terms(terms):
    """
    Finds synonyms for a list of terms using WordNet.
    
    Args:
        terms (list of str): List of terms to find similar terms for.
    
    Returns:
        list of str: Expanded list of terms including synonyms.
    """
    expanded_terms = set(terms)

    for term in terms:
        # Find WordNet synsets for the term
        synsets = wn.synsets(term)
        
        # Extract lemmas (synonyms) from synsets
        synonyms = set(chain.from_iterable([syn.lemma_names() for syn in synsets]))
        
        # Add the synonyms to the expanded terms
        expanded_terms.update(synonyms)

    # Convert to lowercase and remove duplicates
    return list(set(term.lower() for term in expanded_terms))