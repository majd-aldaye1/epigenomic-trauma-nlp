"""
modeling.py

Purpose:
- Analyzes preprocessed text data to extract and categorize terms using machine learning and natural language processing.
- Utilizes Hugging Face's zero-shot classification pipeline for contextual term analysis.
- Categorizes terms into predefined categories (e.g., Mental Health, Epigenetics, Socioeconomic).
- Generates a structured JSON output suitable for further analysis or visualization.

Key Steps:
1. Load preprocessed articles from `preprocessed_articles.json`.
2. Perform term categorization:
   - Use predefined lists and dictionaries to categorize terms into categories like Mental Health, Epigenetics, etc.
   - Augment existing term counts with terms extracted by Hugging Face's zero-shot classification pipeline.
3. Compute relationships:
   - Calculate co-occurrences of terms across categories.
   - Prepare structured data for downstream analysis.
4. Save results to a structured JSON file for further use.

Inputs:
- Preprocessed JSON file (`preprocessed_articles.json`) containing:
  [
      {
          "paper_name": "example.pdf",
          "cleaned_text": "processed text here",
          "term_counts": {
              "mental health terms": {"depression": 5, "anxiety": 3, ...},
              "epigenetic terms": {"methylation": 4, ...},
              ...
          },
          ...
      },
      ...
  ]

Outputs:
- JSON file (`processed_topics.json`) containing:
  [
      {
          "paper_id": 1,
          "categorized_counts": {
              "Mental Health": {"General": {"PTSD": 5, "anxiety": 3, ...}},
              "Epigenetic": {"General": {"DNA methylation": 4, ...}},
              ...
          }
      },
      ...
  ]

How to Use:
- Run this script after `preprocessing.py` to analyze categorized text data.
- Ensure the input file matches the expected format from `preprocessing.py`.
- Use the output JSON file for visualization or further modeling.

Key Features:
- Hugging Face's zero-shot classification for contextual analysis.
- Parallel processing for scalability when handling multiple articles.
- Comprehensive logging and error handling for robust execution.
"""
import logging, time, warnings, json, os, nltk, torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all but critical TensorFlow logs
warnings.filterwarnings("ignore", category=FutureWarning) # Ignore future warnings to keep logs clean
nltk.download('punkt') # Ensure the NLTK tokenizer data is available
from collections import Counter, defaultdict
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from concurrent.futures import ProcessPoolExecutor
from transformers import pipeline 

# Initialize Hugging Face text classification pipeline
logging.info("Initializing Hugging Face pipeline for zero-shot classification...")
device = 0 if torch.cuda.is_available() else -1  # 0: GPU, -1: CPU
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
logging.info("Hugging Face pipeline initialized successfully.")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define categories and subcategories
logging.info("Defining categories and subcategories for classification...")
mental_health_terms = ["depression", "bipolar", "PTSD", "anxiety", "suicide", "generational trauma"]
epigenetic_terms = ["DNA methylation", "BDNF", "SLC6A4", "FKBP5", "OXTR", "stress response", "HPA axis dysregulation"]
ethnographic_terms = {
    "African descent": ["African person", "Black person", "African-American person"],
    "Latino/Hispanic descent": ["Latino person", "Hispanic person", "Latinx community"],
    "Asian descent": ["Asian person", "South Asian person", "East Asian person"],
    "Native American descent": ["Indigenous person", "Native American person", "First Nations person"],
    "Arab descent": ["Arab person", "Middle-Eastern person", "Muslim person"],
    "European descent": ["European person", "Caucasian person", "White person"],
}
socioeconomic_terms = ["low-income", "middle-income", "high-income"]

def categorize_terms(term_counts, categories):
    """
    Categorize terms into defined subcategories.
    Args:
        term_counts (dict): Counts of terms from analysis.
        categories (list or dict): Subcategories for categorization.
    Returns:
        dict: Categorized term counts.
    """
    logging.debug("Starting term categorization...")
    categorized = defaultdict(lambda: defaultdict(int)) # Nested dictionary for counts
    logging.debug(f"Term counts: {term_counts}")

    # Ensure input is a dictionary
    if not isinstance(term_counts, dict):
        logging.error(f"term_counts must be a dictionary. Found {type(term_counts)}")
        return {}
    
    # Handle dictionary structure for categorization
    for term, count in term_counts.items():
        term_added = False    
        if isinstance(categories, dict):  # Handle  nesteddictionary (e.g., ethnographic_terms)
            for category, subterms in categories.items():
                if isinstance(subterms, list) and term in subterms:
                    categorized[category][term] += count
                    term_added = True
                    break
                elif isinstance(subterms, dict):  # Handle nested categories
                    for subcategory, nested_terms in subterms.items():
                        if term in nested_terms:
                            categorized[subcategory][term] += count
                            break
        elif isinstance(categories, list):  # Handle list (e.g., mental_health_terms)
            if term in categories:
                categorized["General"][term] += count
                term_added = True
        
        if not term_added:
            logging.debug(f"Term '{term}' did not match any predefined categories.")

    logging.debug(f"Categorized terms: {dict(categorized)}")
    return dict(categorized)


def analyze_text_with_huggingface(cleaned_text):
    """
    Use Hugging Face pipeline to analyze text for additional terms.
    Args:
        cleaned_text (str): Preprocessed text of the article.
    Returns:
        dict: Additional terms and insights found.
    """
    print("Using device:", "GPU" if device == 0 else "CPU")
    logging.debug("Analyzing text using Hugging Face pipeline...")
    if not cleaned_text.strip():
        logging.warning("Received empty cleaned_text for Hugging Face analysis.")
        return Counter()

    additional_terms = Counter()
    labels = mental_health_terms + epigenetic_terms + list(ethnographic_terms.keys()) + socioeconomic_terms

    # Log and verify chunking
    logging.debug(f"Raw cleaned_text: {cleaned_text}")
    chunks = sent_tokenize(cleaned_text)  # Use sentence tokenization for a single line of text
    if not chunks:
        logging.warning("No valid chunks found after sentence tokenization.")
        return additional_terms
    logging.debug(f"Generated {len(chunks)} chunks: {chunks}")

    # Start processing chunks
    start_time = time.time()
    for chunk in chunks:
        try:
            # Run the classifier on each current chunk
            logging.debug(f"Processing chunk: {chunk}")
            result = classifier(chunk, labels, multi_label=True)  # Note: `multi_label=True` is the correct argument
            logging.debug(f"Result for chunk: {result}")   

            # Add high-confidence terms to the additional terms counter
            for label, score in zip(result["labels"], result["scores"]):
                if score > 0.3:  # Include terms with confidence > 0.3
                    additional_terms[label] += 1
        except Exception as e:
            logging.exception(f"Error processing chunk '{chunk}': {e}")
    logging.info(f"Processing completed for paper in {time.time() - start_time:.2f} seconds")
    logging.debug(f"Final additional terms: {additional_terms}")
    return additional_terms


def process_single_paper(args):
    """
    Process a single paper to categorize terms.
    """
    paper_index, paper = args  # Unpack the arguments
    logging.info(f"Processing paper {paper_index}: {paper.get('paper_name', 'Unknown')}")

    try:
        term_counts = paper.get("term_counts", {})
        cleaned_text = paper.get("cleaned_text", "")

        # Validate term_counts and cleaned_text
        if not isinstance(term_counts, dict):
            logging.warning(f"Invalid or missing term_counts for paper {paper_index}. Skipping.")
            return None
        # if not cleaned_text.strip():
        #     logging.warning(f"Empty cleaned_text for paper {paper_index}. Skipping.")
        #     return None
        
        # Analyze text for additional terms
        additional_terms = analyze_text_with_huggingface(cleaned_text)
        logging.debug(f"Additional terms extracted: {additional_terms}")

        # Merge additional terms with existing term counts
        for category, terms in term_counts.items():
             term_counts[category] = Counter(terms) + additional_terms

        # Categorize terms into subcategories
        categorized_counts = {
            "Mental Health": dict(categorize_terms(term_counts["mental health terms"], mental_health_terms)),
            "Epigenetic": dict(categorize_terms(term_counts["epigenetic terms"], epigenetic_terms)),
            "Socioeconomic": dict(categorize_terms(term_counts["socioeconomic terms"], socioeconomic_terms)),
            "Ethnographic": dict(categorize_terms(term_counts["ethnographic terms"], ethnographic_terms)),
        }
        logging.info(f"Categorized counts for paper {paper_index}: {categorized_counts}")
        logging.info(f"Completed processing paper {paper_index}")
        return {"paper_id": paper_index, "categorized_counts": categorized_counts}

    except Exception as e:
        logging.error(f"Error processing paper {paper_index}: {e}", exc_info=True)
        return None


def process_articles(input_file, output_file):
    logging.info(f"Starting processing of articles: {input_file}")
    with open(input_file, "r") as infile:
        data = json.load(infile)
        logging.debug(f"Loaded input data: {data}")

    papers = data["papers"]
    logging.info(f"Found {len(papers)} papers to process.")

    try:
        # Process in parallel
        logging.info(f"Processing {len(papers)} papers in parallel...")
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(process_single_paper, enumerate(papers, start=1)))
    except Exception as e:
        logging.error(f"Error during multiprocessing: {e}. Falling back to sequential processing.")
        results = [process_single_paper((i, paper)) for i, paper in enumerate(papers, start=1)]

    # Filter out any None results (e.g., skipped papers)
    processed_papers = [result for result in results if result]
    logging.info(f"Successfully processed {len(processed_papers)} papers.")

    # Save the updated JSON
    with open(output_file, "w") as outfile:
        json.dump({"papers": processed_papers}, outfile, indent=4)
        logging.info(f"Processed data saved to {output_file}")
    logging.info("Modeling completed successfully.")

    
