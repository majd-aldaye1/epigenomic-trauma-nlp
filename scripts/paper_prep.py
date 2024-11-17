import spacy
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import json
import pandas as pd
import os
import fitz  # PyMuPDF for PDF text extraction

# Ensure NLTK dependencies are available
nltk.download('punkt')
nltk.download('stopwords')

# Load SciSpacy's biomedical model
nlp = spacy.load('en_core_sci_lg')

# Stopwords: Combining your stopwords and custom stopwords
stop_words = set(stopwords.words('english'))
custom_stopwords = {"study", "https", "http", "gene", "results", "control", "significant", "across", "factor", "health"}
stop_words.update(custom_stopwords)

# Keywords to categorize the abstracts
mental_health_terms = ["depression", "bipolar", "PTSD", "anxiety", "suicide"]
epigenetic_terms = ["methylation", "modification", "aging"]
ethnographic_terms = ["Black", "Latino", "Caucasian", "Asian", "Native American", "Hispanic", "Middle Eastern"]
socioeconomic_terms = ["socioeconomic status", "income inequality", "poverty", "social class", "economic hardship"]

# Preprocessing function to clean text, tokenize, remove stopwords, and lemmatize
def preprocess_text(content):
    if not isinstance(content, str) or content.strip() == "":
        return []  # Return empty list if content is missing or invalid

    # Convert to lowercase
    content = content.lower()
    
    # Remove special characters but keep alphanumeric terms
    content = re.sub(r'[^\w\s]', ' ', content)
    
    # Tokenize the text
    tokens = word_tokenize(content)
    
    # Remove stopwords
    tokens = [token for token in tokens if len(token)>3 and token not in stop_words and not re.search(r'\d', token) and len(token) > 2]
    
    # Apply the biomedical model to the tokens
    doc = nlp(' '.join(tokens))
    
    # Lemmatize tokens and handle scientific entities
    lemmatized_tokens = []
    for token in doc:
        # Keep important terms and entities as is
        if token.text in epigenetic_terms or token.ent_type_ in ['GENE_OR_GENE_PRODUCT', 'CHEMICAL', 'DISEASE']:
            foo.write(f'{token.text}\n')
            lemmatized_tokens.append(token.text)
        else:
            lemmatized_tokens.append(token.lemma_)
    
    return lemmatized_tokens

# Function to categorize content based on predefined terms
def categorize_text(text, keywords):
    
    if not isinstance(text, str):  # Check if the text is valid (not NaN)
        return 'None'
    
    found_keywords = [keyword for keyword in keywords if keyword in text]
    return ', '.join(found_keywords) if found_keywords else 'None'

# Extract text from PDFs in a directory
def extract_pdf_content(pdf_directory):
    pdf_data = []
    for file_name in os.listdir(pdf_directory):
        if file_name.endswith('.pdf'):  # Process only PDF files
            file_path = os.path.join(pdf_directory, file_name)
            print(f"Processing: {file_name}")
            try:
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                pdf_data.append({"File_Name": file_name, "Content": text.strip()})
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
    return pdf_data

# Preprocess the content from PDFs
def preprocess_pdf_content(pdf_data):
    df = pd.DataFrame(pdf_data)
    
    # Apply text preprocessing to the content
    df['Processed_Content'] = df['Content'].apply(preprocess_text)
    
    # Convert tokenized content back to JSON format for storage
    df['Processed_Content'] = df['Processed_Content'].apply(json.dumps)
    
    # Apply categorization based on predefined terms
    df['Mental_Health_Terms'] = df['Content'].apply(lambda x: categorize_text(x, mental_health_terms))
    df['Epigenetic_Terms'] = df['Content'].apply(lambda x: categorize_text(x, epigenetic_terms))
    df['Socioeconomic_Terms'] = df['Content'].apply(lambda x: categorize_text(x, socioeconomic_terms))
    df['Ethnographic_Terms'] = df['Content'].apply(lambda x: categorize_text(x, ethnographic_terms))

    df = df.drop('Content', axis=1)
    
    return df

if __name__ == "__main__":
    foo = open('foo.txt', 'w')
    # Directory containing PDF files
    pdf_directory = 'data/papers'
    
    # Extract content from PDFs
    pdf_data = extract_pdf_content(pdf_directory)
    
    if pdf_data:
        # Preprocess the extracted PDF content
        processed_df = preprocess_pdf_content(pdf_data)
        
        # Save the processed data to a new CSV file
        processed_df.to_csv('data/preprocessed_pdf_content.csv', index=False)
        print("Preprocessing complete and saved to 'data/preprocessed_pdf_content.csv'")
    else:
        print("No valid PDF content extracted.")
    foo.close()
