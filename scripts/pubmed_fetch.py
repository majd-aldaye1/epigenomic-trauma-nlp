# scripts/pubmed_fetch.py
from Bio import Entrez
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

Entrez.email = 'mmm443@cornell.edu'  # Add your email here

# Load PubMedBERT model
model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
print("Model loaded successfully!")

# Function to find synonyms using PubMedBERT embeddings
def get_similar_terms(terms, threshold=0.7):
    term_embeddings = model.encode(terms)
    similarity_matrix = util.cos_sim(term_embeddings, term_embeddings)
    
    # Expand terms based on similarity scores
    expanded_terms = set(terms)
    for i, term in enumerate(terms):
        for j, score in enumerate(similarity_matrix[i]):
            if i != j and score > threshold:  # Use the threshold to find similar terms
                expanded_terms.add(terms[j])
    
    return list(expanded_terms)

def search(query):
    handle = Entrez.esearch(db='pubmed', sort='relevance', retmax='100', retmode='xml', term=query)
    results = Entrez.read(handle)
    return results

def fetch_details(id_list):
    ids = ','.join(id_list)
    handle = Entrez.efetch(db='pubmed', retmode='xml', id=ids)
    results = Entrez.read(handle)
    return results

# Fetch PubMed data with expanded terms
def get_pubmed_data(mental_health_terms, epigenetic_terms, max_results=100):
    # Expand the terms using PubMedBERT
    expanded_mental_health_terms = get_similar_terms(mental_health_terms)
    expanded_epigenetic_terms = get_similar_terms(epigenetic_terms)

    # Build a dynamic query using both expanded term lists
    query = f"({' OR '.join(expanded_mental_health_terms)}) AND ({' OR '.join(expanded_epigenetic_terms)}) AND trauma"
    print(f"Generated Query: {query}")

    # Search PubMed with the generated query
    studies = search(query)
    id_list = studies['IdList']

    # Fetch details for the returned PubMed IDs
    details = fetch_details(id_list)

    # Extract relevant information from the articles
    article_data = []
    for article in details['PubmedArticle']:
        try:
            title = article['MedlineCitation']['Article']['ArticleTitle']
            abstract = article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
            journal = article['MedlineCitation']['Article']['Journal']['Title']
            article_data.append({'Title': title, 'Abstract': abstract, 'Journal': journal})
        except KeyError:
            continue

    # Convert the data to a DataFrame
    df = pd.DataFrame(article_data)
    return df

def save_to_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    # Core mental health and epigenetic terms for querying
    mental_health_terms = ["depression", "bipolar", "PTSD", "anxiety", "suicide"]
    epigenetic_terms = ["DNA methylation", "histone modification", "gene expression"]

    # Fetch data using the expanded terms
    df = get_pubmed_data(mental_health_terms, epigenetic_terms)
    save_to_csv(df, 'data/pubmed_articles.csv')
