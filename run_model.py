import json
import warnings
from modeling import process_articles

# Suppress deprecation warnings
warnings.filterwarnings("ignore", message="The `multi_class` argument has been deprecated")

def main():
    input_file = "test_1.json"
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
