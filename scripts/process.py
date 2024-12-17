"""
preprocess.py

Purpose:
- Preprocess raw PDF articles for downstream natural language processing (NLP) analysis.
- Performs text extraction, cleaning, tokenization, lemmatization, and categorization.
- Calculates co-occurrence matrices and Jaccard similarity scores between predefined categories.
- Generates structured outputs for insights into relationships between trauma, mental health, and epigenetic terms.

Key Steps:
1. **Load Predefined Terms**: Load expanded terms for predefined categories (e.g., mental health, epigenetics) from a JSON file (`expanded_terms.json`).
2. **Text Extraction**: Extract raw text content from PDF files using PyMuPDF.
3. **Text Preprocessing**:
   - Clean text (remove links, standalone numbers, punctuation, and redundant spaces).
   - Lemmatize and filter tokens using SpaCy's transformer-based model (`en_core_web_trf`).
   - Extract NORP (ethnicity) entities to identify demographic mentions.
4. **Categorize Terms**:
   - Match tokens to predefined terms and count their occurrences across categories.
   - Compute term co-occurrence matrices to identify relationships.
5. **Calculate Global Statistics**:
   - Generate Jaccard similarity scores and co-occurrence counts between term categories.
   - Track metadata related to ethnicity and socioeconomic mentions.
6. **Output Results**: Save results to a structured JSON file (`preprocessed_articles.json`) for analysis.

Inputs:
- Directory containing raw PDF articles (`data/papers`).
- JSON file of expanded terms (`expanded_terms.json`) generated or curated earlier.

Outputs:
- JSON file (`preprocessed_articles.json`) containing:
    - Extracted, cleaned, and tokenized text for each paper.
    - Term counts categorized into predefined groups (e.g., mental health, epigenetics).
    - Co-occurrence matrices showing relationships between term categories.
    - Jaccard similarity scores for inter-category overlaps.
    - Metadata on disparities, including ethnicity and socioeconomic mentions.

Sample JSON Output Structure:
[
    {
        "paper_name": "example.pdf",
        "cleaned_text": "processed lemmatized text here",
        "term_counts": {
            "mental health terms": {"depression": 5, "anxiety": 3},
            "epigenetic terms": {"methylation": 4, "CpG islands": 2}
        },
        "co_occurrence_matrix": {
            "mental health terms": {"methylation": 3, "CpG islands": 2}
        },
        "disparity_metadata": {
            "ethnicity": "african descent",
            "socioeconomic_status": "low-income"
        }
    },
    ...
]

How to Use:
1. Ensure `expanded_terms.json` is available with relevant expanded categories and terms.
2. Place raw PDFs in the `data/papers` directory.
3. Run this script to preprocess the articles:
    $ python preprocess.py
4. Use the output JSON file for further analysis in downstream tasks such as topic modeling or visualization.

Dependencies:
- PyMuPDF (text extraction from PDFs)
- SpaCy (`en_core_web_trf` for tokenization and NER)
- NLTK (stopwords removal)
- JSON, re, and logging libraries for processing and logging.

Notes:
- NORP entities (e.g., ethnic groups) are extracted and included under 'ethnographic terms'.
- Co-occurrence and Jaccard similarity calculations provide insights into inter-category relationships.
"""

import os, re, json, logging, fitz, spacy
from nltk.corpus import stopwords
from nltk import download
from collections import defaultdict, Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK dependencies
download("punkt")
download("stopwords")

# Load SciSpacy's biomedical model
# nlp = spacy.load("en_core_sci_lg")
logging.info("Loading SpaCy model...")
try:
    nlp = spacy.load("en_core_web_trf")
    logging.info("SpaCy model loaded successfully.")
except Exception as e:
    logging.error("Error loading SpaCy model. Ensure the model is installed. Use 'python -m spacy download en_core_web_trf'.", exc_info=True)
    raise e

# Paths relative to the script's directory
RAW_ARTICLES_DIR = "./data/papers"
OUTPUT_FILE = "./preprocessed_articles.json"  # Adjust relative path
TOP_TERMS_FILE = './expanded_terms.json'

print(f"RAW_ARTICLES_DIR: {RAW_ARTICLES_DIR}")
print(f"TOP_TERMS_FILE: {TOP_TERMS_FILE}")
print(f"OUTPUT_FILE: {OUTPUT_FILE}")

# Load expanded terms (core + similar terms)
try:
    with open(TOP_TERMS_FILE, "r", encoding="ascii", errors="replace") as file:
        expanded_terms = json.load(file)
    logging.info("Successfully loaded expanded terms.")
except FileNotFoundError as e:
    logging.error(f"Error: {e}. Ensure that the file exists at: {TOP_TERMS_FILE}", exc_info=True)
    raise e


def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF files.
    Args:
        pdf_path (str): File containing PDF content.
    Returns:
        list: List of dictionaries with extracted content.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        logging.info(f"Successfully extracted text of length {len(text)} from {pdf_path}")
        return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}", exc_info=True)
        return ""


def clean_text(text):
    """
    Perform basic text cleaning and tokenization.
    Args:
        text (str): Raw text to clean.
    Returns:
        str: Cleaned text.
    """
    logging.debug("Cleaning text...")
    # Remove links starting with 'http'
    text = re.sub(r"http\S+", "", text)
    
    # Remove patterns like '[#]'
    text = re.sub(r"\[\d+\]", "", text)

    # Remove all mentions of DOI, pubmed, and any common paper section titles
    text = re.sub(r"(DOI|pubmed|review|abstract|keywords|review|download|reference|acknowledge)", "", text, flags=re.I)

    # Remove standalone numbers
    text = re.sub(r"\b\d+\b", "", text)
    
    # Remove standalone letters
    text = re.sub(r"\b\w\w\b", "", text)

    # Remove standalone letters
    text = re.sub(r"\b\w\b", "", text)

    # Remove standalone punctuation
    text = re.sub(r"[^a-zA-Z\d\s:]", "", text)  
    
    # Remove extra spaces created by the removals
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_and_process(doc):
    """
    Lemmatize text using SciSpacy and preserve biomedical entities.
    Args:
        text (str): Cleaned text to process.
    Returns:
        list: Lemmatized tokens.
    """
    logging.debug("Starting lemmatization and filtering...")
    # Remove prounouns, determiners, articles, wh-words, etc.
    tags_to_remove = ['$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'CC', 'CD',
                       'DT', 'EX', 'FW', 'HYPH', 'IN', 'LS', 'MD',
                         'NFP', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'RBR', 'RBS', 'RB',
                           'SYM', 'TO', 'UH', 'JJS', 'JJR'
                         'WDT', 'WP', 'WP$', 'XX', '_SP', '``'
                         ]
    
    # Remove irrelevant entities
    ents_to_remove  = ["TIME", "DATE", "GPE", "PERSON", "FAC", "MONEY", "ORG", 'ORDINAL', 'PERCENT',
                       'LAW', 'EVENT', 'QUANTITY', 'PRODUCT', 'CARDINAL', 'WORK_OF_ART', 'LOC']
                       
    cleaned_text = []
    for token in doc:
        if token.tag_ in tags_to_remove or token.ent_type_ in ents_to_remove or token.text.lower() in stopwords.words("English"):
            # Print out removed tokens for debugging
            print(token.text,token.tag_, token.ent_type_, file = f)
        elif token.ent_type_ == "" and token.tag_ not in tags_to_remove:
            cleaned_text.append(token.lemma_)
        elif token.ent_type_ not in ents_to_remove and token.tag_ not in tags_to_remove:
            cleaned_text.append(token.text)
    logging.debug(f"Finished lemmatization. Total tokens: {len(cleaned_text)}")
    return cleaned_text


def categorize_terms(text, expanded_terms):
    """
    Categorize text using expanded terms and count term frequencies.
    Args:
        text (str): Processed text to analyze.
        expanded_terms (dict): Dictionary of core terms and their expanded similar terms.
    Returns:
        dict: Term counts for each category.
    """
    logging.debug("Categorizing terms...")
    term_counts = {category: Counter() for category in expanded_terms.keys()}
    tokens = text.split()
    for category, terms in expanded_terms.items():
        term_counts[category] = Counter(t for t in tokens if t in terms)
    logging.debug(f"Term counts: {term_counts}")
    return term_counts


def compute_co_occurrence(term_counts):
    """Compute a co-occurrence matrix for terms across categories."""
    co_occurrence_matrix = defaultdict(lambda: defaultdict(int))
    for category1, terms1 in term_counts.items():
        for category2, terms2 in term_counts.items():
            if category1 != category2 and len(terms1) > 0 and len(terms2) > 0:
                # Count each pair of terms that appear
                for term1 in terms1:
                    for term2 in terms2:
                        co_occurrence_matrix[category1][term2] += terms1[term1]
    return co_occurrence_matrix



def extract_norp_entities(doc):
    """
    Extract NORP entities from the processed text using NER.
    
    Args:
        doc (spacy.tokens.Doc): A processed SpaCy Doc object.
        
    Returns:
        list: List of NORP entities found in the text.
    """
    norp_entities = [ent.text for ent in doc.ents if ent.label_ == "NORP"]
    return norp_entities


def preprocess_articles(input_dir=RAW_ARTICLES_DIR, expanded_terms=expanded_terms):
    """
    Process raw articles (PDFs only) to clean, categorize, and calculate statistics.
    Args:
        input_dir (str): Directory containing raw articles.
        expanded_terms (dict): Dictionary with core terms and expanded similar terms.
    Returns:
        list: List of processed article metadata.
    """
    logging.info("Starting article preprocessing...")
    processed_articles = []
    global_term_counts = {category: Counter() for category in expanded_terms.keys()}
    global_relationships = defaultdict(lambda: {"terms": Counter(), "co_occurrence_count": 0, "jaccard_similarity": 0.0})
    MAX_NLP_LENGTH = 50000  # Limit for NLP processing

    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        
        # Process only PDF files
        if not file_name.endswith(".pdf"):
            logging.info(f"Skipping non-PDF file: {file_name}")
            continue

        logging.info(f"Processing file: {file_name}")

        try: 
            # Validate PDF
            try:
                with fitz.open(file_path) as doc:
                    pass
            except Exception as e:
                logging.error(f"Invalid or corrupted PDF: {file_name} - {e}")
                continue

            # Extract text from the PDF
            raw_text = extract_text_from_pdf(file_path)

            if not raw_text.strip():
                logging.warning(f"No extractable text found in file: {file_name}")
                continue

            # Clean, tokenize, and categorize terms
            cleaned_text = clean_text(raw_text)
            truncated_text = cleaned_text[:MAX_NLP_LENGTH]

            # Perform NLP processing with error handling
            try:
                doc = nlp(truncated_text)
            except Exception as e:
                logging.error(f"Error during NLP processing for {file_name}: {e}")
                continue
        
            lemmatized_text = " ".join(lemmatize_and_process(doc))
            
            # Extract NORP entities
            norp_entities = extract_norp_entities(doc)

            # Categorize terms (include NORP entities under ethnographic terms)
            term_counts = categorize_terms(lemmatized_text, expanded_terms)
            term_counts["ethnographic terms"].update(norp_entities)

            # Update global term counts
            for category, counts in term_counts.items():
                global_term_counts[category].update(counts)

            # Compute co-occurrence matrix
            co_occurrence_matrix = compute_co_occurrence(term_counts)

            # Add metadata (optional step based on text heuristics)
            disparity_metadata = {
                "ethnicity": next((term for term in expanded_terms["ethnographic terms"] if term in lemmatized_text), "Unknown"),
                "socioeconomic_status": next((term for term in expanded_terms["socioeconomic terms"] if term in lemmatized_text), "Unknown"),
            }

            logging.info(f"Disparity metadata for {file_name}: {disparity_metadata}")

            processed_articles.append({
                "paper_name": file_name,
                "cleaned_text": lemmatized_text,
                "term_counts": term_counts,
                "co_occurrence_matrix": co_occurrence_matrix,
                "disparity_metadata": disparity_metadata
            })

        except Exception as e:
            logging.error(f"Error processing file {file_name}: {e}", exc_info=True)
            continue

    # Calculate Jaccard similarities and global relationships
    for category1, terms1 in global_term_counts.items():
        for category2, terms2 in global_term_counts.items():
            if category1 != category2:
                intersection = sum((terms1 & terms2).values())
                union = sum((terms1 | terms2).values())
                global_relationships[(category1, category2)]["jaccard_similarity"] = intersection / union if union > 0 else 0.0
                global_relationships[(category1, category2)]["co_occurrence_count"] += intersection

    # Inline conversion of tuple keys to strings
    def convert_tuple_keys(d):
        """Recursively convert tuple keys to strings in a nested dictionary."""
        if not isinstance(d, dict):
            return d
        return {str(k): convert_tuple_keys(v) for k, v in d.items()}
    
    # Convert tuple keys to strings
    global_relationships_str_keys = convert_tuple_keys(global_relationships)

    # Save processed data
    with open(OUTPUT_FILE, "w") as output_file:
        json.dump({
            "papers": processed_articles,
            "global_summary": {
                "total_term_counts": {k: dict(v) for k, v in global_term_counts.items()},
                "top_relationships": global_relationships_str_keys
            }
        }, output_file, indent=4)

    logging.info(f"Processed articles saved to {OUTPUT_FILE}")
    return processed_articles



if __name__ == "__main__":
    processed_articles = preprocess_articles()
    logging.info(f"Processed {len(processed_articles)} articles successfully.")