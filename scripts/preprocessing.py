# Modified preprocessing.py to account for epigenetic terms
import spacy
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import json
import pandas as pd

# Ensure 'punkt' tokenizer is available
nltk.download('punkt')
nltk.download('punkt_tab')

# Load SciSpacy's biomedical model
nlp = spacy.load('en_core_sci_lg')

# Stopwords: Combining your stopwords and their custom stopwords
stop_words = set(stopwords.words('english'))
custom_stopwords = {"study", "results", "control", "significant"}
stop_words.update(custom_stopwords)

# Keywords from your code to categorize the abstracts
mental_health_terms = ["depression", "bipolar", "PTSD", "anxiety", "suicide"]
epigenetic_terms = ["DNA methylation", "histone modification", "gene expression"]  
ethnographic_terms = ["race", "ethnicity", "African American", "Latino", "Caucasian", "Asian", "Native American", "Hispanic", "Indigenous", "Arab", "Middle Eastern"]
socioeconomic_terms = ["socioeconomic status", "income inequality", "poverty", "social class", "education disparity", "economic hardship"]

# Preprocessing function to clean the text, tokenize, remove stopwords, and lemmatize
def preprocess_text(abstract):
    # Convert to lowercase
    abstract = abstract.lower()
    
    # Remove punctuation and special characters
    abstract = re.sub(r'\W+', ' ', abstract)
    
    # Tokenize the text
    tokens = word_tokenize(abstract)
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # Apply the biomedical model to the tokens
    doc = nlp(' '.join(tokens))
    
    # Lemmatize tokens and handle scientific entities
    lemmatized_tokens = []
    for token in doc:
        # Check if token is one of the important epigenetic terms
        if token.text in epigenetic_terms:
            # Keep the term as it is, without lemmatization
            lemmatized_tokens.append(token.text)
        # Check if token is a recognized scientific entity
        elif token.ent_type_ in ['GENE_OR_GENE_PRODUCT', 'CHEMICAL', 'DISEASE']:
            # Keep the token as it is if it's a scientific entity
            lemmatized_tokens.append(token.text)
        else:
            # Apply normal lemmatization for non-entities
            lemmatized_tokens.append(token.lemma_)
    
    return lemmatized_tokens

# Function to categorize abstracts based on predefined terms
def categorize_text(text, keywords):
    found_keywords = [keyword for keyword in keywords if keyword.lower() in text.lower()]
    return ', '.join(found_keywords) if found_keywords else 'None'

# Preprocess abstracts: Tokenize, lemmatize, and categorize by your four dimensions
def preprocess_abstracts(df):
    # Apply text preprocessing to the abstracts
    df['Processed_Abstract'] = df['Abstract'].apply(preprocess_text)
    
    # Convert tokenized abstracts back to JSON format for storage (keeping their use of json.dumps)
    df['Processed_Abstract'] = df['Processed_Abstract'].apply(json.dumps)
    
    # Apply categorization based on predefined terms
    df['Mental_Health_Terms'] = df['Abstract'].apply(lambda x: categorize_text(x, mental_health_terms))
    df['Epigenetic_Terms'] = df['Abstract'].apply(lambda x: categorize_text(x, epigenetic_terms))
    df['Socioeconomic_Terms'] = df['Abstract'].apply(lambda x: categorize_text(x, socioeconomic_terms))
    df['Ethnographic_Terms'] = df['Abstract'].apply(lambda x: categorize_text(x, ethnographic_terms))
    
    return df

if __name__ == "__main__":
    # Load the raw data from the CSV file
    df = pd.read_csv('data/pubmed_articles.csv')
    
    # Preprocess the abstracts and categorize them
    df = preprocess_abstracts(df)
    
    # Save the processed DataFrame to a new CSV file
    df.to_csv('data/preprocessed_pubmed_articles.csv', index=False)
    print("Preprocessing complete and saved to 'data/preprocessed_pubmed_articles.csv'")
