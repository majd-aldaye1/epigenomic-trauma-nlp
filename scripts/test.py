import numpy as np
from gensim.models import KeyedVectors
from gensim import downloader as api

# Load the FastText model
word_vectors = api.load('fasttext-wiki-news-subwords-300')

def get_phrase_embedding(phrase, model):
    # FastText can handle phrases directly
    return model[phrase]

def generate_similar_terms(input_terms, model, topn=10):
    similar_terms = set(input_terms)
    
    for term in input_terms:
        try:
            term_vector = model[term]
        except KeyError:
            term_vector = get_phrase_embedding(term, model)
        similar_words = model.similar_by_vector(term_vector, topn=topn)
        for word, score in similar_words:
            similar_terms.add(word)
            print(word)
    return list(similar_terms)

def expand_and_print_terms(input_terms, model, list_name, topn=10):
    expanded_terms = generate_similar_terms(input_terms, model, topn)
    expanded_terms.sort()
    print(f"{list_name} = [")
    for term in expanded_terms:
        print(f'    "{term}",')
    print("]\n")

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

# Expand and print terms for each list
expand_and_print_terms(mental_health_terms, word_vectors, "mental_health_terms", topn=10)
expand_and_print_terms(epigenetic_terms, word_vectors, "epigenetic_terms", topn=10)
expand_and_print_terms(ethnographic_terms, word_vectors, "ethnographic_terms", topn=10)
expand_and_print_terms(socioeconomic_terms, word_vectors, "socioeconomic_terms", topn=10)
