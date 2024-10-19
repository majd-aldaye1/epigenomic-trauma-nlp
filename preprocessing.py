# scripts/preprocessing.py
import spacy
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Ensure 'punkt' tokenizer is available
nltk.download('punkt')
nltk.download('punkt_tab')

# Load SciSpacy's biomedical model
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

def preprocess_text(abstract):
    # Convert to lowercase
    abstract = abstract.lower()
    # Remove punctuation and special characters
    abstract = re.sub(r'\W+', ' ', abstract)
    # Tokenize the text
    tokens = word_tokenize(abstract)
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize tokens
    doc = nlp(' '.join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]
    return lemmatized_tokens

def preprocess_abstracts(df):
    df['Processed_Abstract'] = df['Abstract'].apply(preprocess_text)
    return df

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('data/pubmed_articles.csv')
    df = preprocess_abstracts(df)
    df.to_csv('data/preprocessed_pubmed_articles.csv', index=False)
    print("Preprocessing complete and saved to 'data/preprocessed_pubmed_articles.csv'")
