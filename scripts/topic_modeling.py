import gensim
import pandas as pd
from gensim import corpora
import json
import logging
import os
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to build the LDA model
def build_topic_model(processed_texts, num_topics=5, passes=100):
    # logging.info("Building dictionary from processed texts...")
    dictionary = corpora.Dictionary(processed_texts)

    # logging.info("Creating corpus (bag-of-words representation)...")
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # logging.info(f"Training LDA model with {num_topics} topics and {passes} passes...")
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
    logging.info("LDA model training complete!")
    
    return lda_model, dictionary, corpus

# Function to print topics from the LDA model
def print_topics(lda_model, num_words=20):
    # logging.info(f"Displaying the top {num_words} words from each topic:")
    topics = lda_model.print_topics(num_words=num_words)
    for topic in topics:
        print(topic)

# Function to save the model, dictionary, and corpus for later use
def save_model(lda_model, dictionary, corpus, model_path='models'):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logging.info(f"Saving LDA model to {model_path}/lda_model...")
    lda_model.save(f'{model_path}/lda_model')
    logging.info(f"Saving dictionary to {model_path}/dictionary.dict...")
    dictionary.save(f'{model_path}/dictionary.dict')
    logging.info(f"Saving corpus to {model_path}/corpus.mm...")
    corpora.MmCorpus.serialize(f'{model_path}/corpus.mm', corpus)
    logging.info("Model, dictionary, and corpus saved successfully!")

# Function to visualize the LDA model using pyLDAvis
def visualize_topics(lda_model, corpus, dictionary):
    logging.info("Visualizing LDA model with pyLDAvis...")
    lda_display = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_display, 'models/lda_visualization.html')
    logging.info("LDA model visualization saved as 'models/lda_visualization.html'. Open this file in a browser to view it.")

if __name__ == "__main__":
    logging.info("Loading preprocessed PubMed articles from CSV...")
    
    try:
        # Ensure correct column names
        column_names = ['Title', 'Abstract', 'Journal', 'Processed_Abstract', 
                        'Mental_Health_Terms', 'Epigenetic_Terms', 
                        'Socioeconomic_Terms', 'Ethnographic_Terms']
        
        # Load the CSV with correct headers
        df = pd.read_csv('data/preprocessed_pubmed_articles.csv', header=None, names=column_names)
    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
        exit()

    # Check if the 'Processed_Abstract' column exists and handle missing data
    if 'Processed_Abstract' in df.columns:
        logging.info("Converting JSON-encoded strings back to token lists...")

        def safe_json_loads(x):
            try:
                if pd.notnull(x) and x.strip():  # Check if not null and not empty
                    return json.loads(x)
                else:
                    return []  # Return empty list if it's invalid or empty
            except json.JSONDecodeError:
                logging.warning(f"Failed to decode JSON for row: {x}")
                return []  # Return empty list if decoding fails

        df['Processed_Abstract'] = df['Processed_Abstract'].apply(safe_json_loads)
    else:
        logging.error("Processed_Abstract column is missing in the CSV.")
        exit()

    # Verify that each row in Processed_Abstract is a list of tokens
    assert all(isinstance(row, list) for row in df['Processed_Abstract']), "Each processed abstract must be a list of tokens"
    
    # Extract tokenized texts as a list for Gensim
    all_tokens = df['Processed_Abstract'].tolist()

    # Build the LDA model
    lda_model, dictionary, corpus = build_topic_model(all_tokens, num_topics=5, passes=20)

    # Print the top 15 words from each topic
    print_topics(lda_model)

    # Save the model, dictionary, and corpus
    save_model(lda_model, dictionary, corpus, model_path='models')

    # Optionally, visualize the LDA model using pyLDAvis
    visualize_topics(lda_model, corpus, dictionary)
