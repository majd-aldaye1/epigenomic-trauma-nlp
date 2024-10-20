import logging
from Bio import Entrez
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import time

# Setup logging (this should be set up early)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set Entrez email
Entrez.email = 'mmm443@cornell.edu'  # Add your email here

# Load PubMedBERT model
model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
logging.info("Model loaded successfully!")

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

# Search PubMed
def search(query, retstart=0, retmax=100):
    try:
        handle = Entrez.esearch(db='pubmed', sort='relevance', retstart=retstart, retmax=retmax, retmode='xml', term=query)
        results = Entrez.read(handle)
        return results
    except Exception as e:
        logging.error(f"Error fetching PubMed search results: {e}")
        return None

# Fetch PubMed article details
def fetch_details(id_list):
    try:
        ids = ','.join(id_list)
        handle = Entrez.efetch(db='pubmed', retmode='xml', id=ids)
        results = Entrez.read(handle)
        return results
    except Exception as e:
        logging.error(f"Error fetching PubMed details: {e}")
        return None

# Fetch PubMed data with expanded terms
def get_pubmed_data(mental_health_terms, epigenetic_terms, ethnographic_terms, socioeconomic_terms, max_results=100):
    # Expand the terms using PubMedBERT
    expanded_mental_health_terms = get_similar_terms(mental_health_terms)
    expanded_epigenetic_terms = get_similar_terms(epigenetic_terms)
    expanded_ethnographic_terms = get_similar_terms(ethnographic_terms)
    expanded_socioeconomic_terms = get_similar_terms(socioeconomic_terms)

    # Build a dynamic query using all expanded term lists
    query = (
        f"({' OR '.join(expanded_mental_health_terms)}) AND "
        f"({' OR '.join(expanded_epigenetic_terms)}) OR "
        f"({' OR '.join(expanded_ethnographic_terms)}) OR "
        f"({' OR '.join(expanded_socioeconomic_terms)})"
    )

    logging.info(f"Generated Query: {query}")

    all_article_data = []
    retstart = 0
    while len(all_article_data) < max_results:
        studies = search(query, retstart=retstart)
        if studies is None:
            break  # Exit loop if there's an error

        id_list = studies.get('IdList', [])
        if not id_list:
            logging.info(f"No more results found at retstart={retstart}.")
            break  # No more results to fetch

        details = fetch_details(id_list)
        if details is None:
            break  # Exit loop if there's an error

        # Extract relevant information from the articles
        for article in details['PubmedArticle']:
            try:
                title = article['MedlineCitation']['Article']['ArticleTitle']
                abstract = article['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', [''])[0]
                journal = article['MedlineCitation']['Article']['Journal']['Title']
                all_article_data.append({'Title': title, 'Abstract': abstract, 'Journal': journal})
            except KeyError:
                logging.warning(f"Missing expected fields in article: {article}")
                continue

        # Increment retstart to fetch next batch of results
        retstart += len(id_list)
        time.sleep(1)  # Be mindful of API rate limits

        # Break if we exceed max_results
        if len(all_article_data) >= max_results:
            logging.info(f"Reached max_results limit of {max_results}.")
            break

    # Convert the data to a DataFrame, ensuring it only contains max_results entries
    df = pd.DataFrame(all_article_data[:max_results])
    return df

def save_to_csv(df, filename):
    df.to_csv(filename, index=False)
    logging.info(f"Data saved to {filename}")

if __name__ == "__main__":
    # Core mental health and epigenetic terms for querying
    mental_health_terms = ["depression", "bipolar", "PTSD", "anxiety", "suicide"]
    epigenetic_terms = ["DNA methylation", "histone modification", "gene expression"]

    # New terms for race/ethnicity and socioeconomic disparities
    ethnographic_terms = ["race", "ethnicity", "African American", "Latino", "Caucasian", "Asian", "Native American", "Hispanic", "Indigenous", "Arab", "Middle Eastern"]
    socioeconomic_terms = ["socioeconomic status", "income inequality", "poverty", "social class", "education disparity", "economic hardship"]

    # Fetch data using the expanded terms
    df = get_pubmed_data(mental_health_terms, epigenetic_terms, ethnographic_terms, socioeconomic_terms, max_results=500)
    save_to_csv(df, 'data/pubmed_articles.csv')
