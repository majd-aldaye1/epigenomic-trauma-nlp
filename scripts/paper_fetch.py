import logging
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import subprocess
import os


os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Generate the query
def generate_query(mental_health_terms, epigenetic_terms, ethnographic_terms, socioeconomic_terms):
    expanded_mental_health_terms = get_similar_terms(mental_health_terms)
    expanded_epigenetic_terms = get_similar_terms(epigenetic_terms)
    expanded_ethnographic_terms = get_similar_terms(ethnographic_terms)
    expanded_socioeconomic_terms = get_similar_terms(socioeconomic_terms)

    # Build a dynamic query using all expanded term lists expanded_epigenetic_terms
    query = (
        f"({' OR '.join(expanded_epigenetic_terms)}) AND "
        f"({' OR '.join(expanded_mental_health_terms)}) OR "
        f"({' OR '.join(expanded_ethnographic_terms)}) OR "
        f"({' OR '.join(expanded_socioeconomic_terms)})"
    )
    logging.info(f"Generated Query: {query}")
    return query

# Fetch papers using PyPaperBot
def fetch_papers(query, scholar_pages=1, min_year=2000, output_dir="./data/papers"):
    # Ensure output directory exists
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
        # Execute the command
        subprocess.run(command, check=True)
        logging.info(f"Papers successfully fetched and saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error fetching papers with PyPaperBot: {e}")

if __name__ == "__main__":
    # Core mental health and epigenetic terms for querying
    mental_health_terms = ["depression", "bipolar", "PTSD", "anxiety", "suicide"]
    epigenetic_terms = ["DNA methylation", "histone modification", "gene expression"]

    # New terms for race/ethnicity and socioeconomic disparities
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
