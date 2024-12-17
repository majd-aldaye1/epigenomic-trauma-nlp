import json
import warnings
from modeling import process_articles
import nltk
import nltk
import os

# Explicitly set the NLTK data path
nltk_data_path = r"C:\Users\snedm\Documents\Cornell\2024 Fall\CS 4701\cs4701\final_demo\CAP_Epigenomics-Analysis_ma798_mmm443\nltk_data"
if os.path.exists(nltk_data_path):
    nltk.data.path.append(nltk_data_path)
    print(f"NLTK data path set to: {nltk_data_path}")
else:
    print(f"Error: Specified NLTK path does not exist: {nltk_data_path}")

# Test loading 'punkt'
try:
    from nltk.tokenize import sent_tokenize
    test_sentence = "This is a test. Let's see if it works!"
    print(sent_tokenize(test_sentence))  # Should tokenize successfully
    print("NLTK 'punkt' loaded successfully!")
except Exception as e:
    print(f"Error loading 'punkt': {e}")

# Suppress deprecation warnings
warnings.filterwarnings("ignore", message="The `multi_class` argument has been deprecated")

def main():
    input_file = "small_preprocessed.json"
    output_file = "output_file.json"

    # Run the process_articles function
    process_articles(input_file, output_file)

    # Confirm processing
    with open(output_file, "r") as f:
        data = json.load(f)
        print(json.dumps(data, indent=4))  # Print output for verification

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # For Windows multiprocessing
    main()
