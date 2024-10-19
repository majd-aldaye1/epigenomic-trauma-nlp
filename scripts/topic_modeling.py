import gensim
from gensim import corpora
import json
import logging
import os
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_ngrams(tokenized_texts):
    """Generate bigrams and trigrams from tokenized texts."""
    bigram = Phrases(tokenized_texts, min_count=3, threshold=3)
    trigram = Phrases(bigram[tokenized_texts], threshold=3)

    bigram_phraser = Phraser(bigram)
    trigram_phraser = Phraser(trigram)

    return [trigram_phraser[bigram_phraser[text]] for text in tokenized_texts]


def apply_tfidf(corpus):
    """Apply TF-IDF weighting to the corpus."""
    # Convert list of lists into a joined string format for TF-IDF
    docs_as_texts = [' '.join(doc) for doc in corpus]
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs_as_texts)
    
    # Return the TF-IDF weighted corpus
    return tfidf_matrix, tfidf_vectorizer

# Build topic model with optimized hyperparameters using Bag-of-Words
def build_topic_model(processed_texts, num_topics=10, passes=15):
    """Build the LDA model based on the tokenized, processed texts."""
    # Create a dictionary from the preprocessed text
    dictionary = corpora.Dictionary(processed_texts)
    dictionary.filter_extremes(no_above=0.9)

    # Create a corpus: a bag-of-words representation of the texts
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # Build LDA model with tuned hyperparameters
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus, num_topics=num_topics, id2word=dictionary, 
        passes=passes, alpha='auto', eta='auto'
    )

    return lda_model, dictionary, corpus

def print_topics(lda_model, num_words=5):
    topics = lda_model.print_topics(num_words=num_words)
    for topic in topics:
        print(topic)

# Function to visualize the LDA model using pyLDAvis
def visualize_topics(lda_model, corpus, dictionary):
    logging.info("Visualizing LDA model with pyLDAvis...")
    lda_display = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_display, 'lda_visualization.html')
    logging.info("LDA model visualization saved as 'lda_visualization.html'. Open this file in a browser to view it.")

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('data/preprocessed_pubmed_articles.csv')
    
    # Convert the JSON-encoded strings back to lists
    df['Processed_Abstract'] = df['Processed_Abstract'].apply(json.loads)
    
    # Tokenized data
    all_tokens = df['Processed_Abstract'].tolist()
    
    # 1. Generate n-grams (bigrams and trigrams)
    tokenized_texts_with_ngrams = generate_ngrams(all_tokens)
    
    # 2. Apply TF-IDF
    tfidf_matrix, tfidf_vectorizer = apply_tfidf(tokenized_texts_with_ngrams)
    
    # Use the tokenized texts with n-grams for LDA model creation
    lda_model, dictionary, corpus = build_topic_model(tokenized_texts_with_ngrams, num_topics=5, passes=20)
    print_topics(lda_model)
    visualize_topics(lda_model, corpus, dictionary)
    