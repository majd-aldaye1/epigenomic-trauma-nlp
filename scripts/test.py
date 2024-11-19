import numpy as np
from gensim.models import KeyedVectors
from gensim import downloader as api

# Load the FastText model (approx. 7GB RAM required)
word_vectors = api.load('fasttext-wiki-news-subwords-300')

def get_phrase_embedding(phrase, model):
    # First, attempt to get the embedding of the phrase as a whole
    try:
        embedding = model[phrase]
        return embedding
    except KeyError:
        # If the phrase is not in the vocabulary, compute the mean of its words
        words = phrase.split()
        embeddings = []
        for word in words:
            # Attempt to get the word's embedding
            try:
                embedding = model[word]
            except KeyError:
                # Use subword embedding for OOV words
                embedding = model.get_vector(word, norm=True)
            embeddings.append(embedding)
        if embeddings:
            # Compute the average embedding
            return np.mean(embeddings, axis=0)
        else:
            return None  # No valid embeddings found

def generate_similar_terms(input_terms, model, topn=15):
    term_similars = {}
    
    for term in input_terms:
        # Get the term's embedding
        term_vector = get_phrase_embedding(term, model)
        if term_vector is None:
            print(f"No valid embeddings found for '{term}', skipping.")
            continue
        # Get the most similar words/phrases
        similar_words = model.similar_by_vector(term_vector, topn=topn)
        similar_terms = [word for word, score in similar_words]
        term_similars[term] = similar_terms
    return term_similars

def expand_and_print_terms(input_terms, model, list_name):
    term_similars = generate_similar_terms(input_terms, model)
    print(f"{list_name}:", file=out_f)
    for term, similars in term_similars.items():
        similar_terms_str = ', '.join(similars)
        print(f"{term}: {similar_terms_str}", file=out_f)
    print("\n", file=out_f)

# Sample Inputs
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

# Open the output file for writing
with open('similar_terms.txt', 'w', encoding='utf-8') as out_f:
    # Expand and print terms for each list
    expand_and_print_terms(mental_health_terms, word_vectors, "Mental Health Terms")
    expand_and_print_terms(epigenetic_terms, word_vectors, "Epigenetic Terms")
    expand_and_print_terms(ethnographic_terms, word_vectors, "Ethnographic Terms")
    expand_and_print_terms(socioeconomic_terms, word_vectors, "Socioeconomic Terms")
