import unittest
import json
import logging
import os
import sys
print("Current directory:", os.getcwd())


# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Log current working directory
logging.debug("Current Working Directory: %s", os.getcwd())

# Update sys.path to ensure modeling.py is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

try:
    from modeling import process_articles, categorize_terms
    logging.debug("Successfully imported `process_articles` and `categorize_terms`.")
except ImportError as e:
    logging.error("Failed to import `modeling`: %s", e)
    raise

class TestTopicModeling(unittest.TestCase):
    def setUp(self):
        """
        Set up test data and file paths.
        """
        logging.debug("Setting up test paths and files.")
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.input_file = os.path.join(base_path, "modeling_tests", "test_1.json")
        self.expected_file = os.path.join(base_path, "modeling_tests", "expected_1.json")
        self.output_file = os.path.join(base_path, "modeling_tests", "output_1.json")
        
        # Ensure the modeling_tests directory exists
        modeling_tests_path = os.path.join(base_path, "modeling_tests")
        if not os.path.exists(modeling_tests_path):
            os.makedirs(modeling_tests_path)
            print(f"Created directory: {modeling_tests_path}")
            
        logging.debug("Input file: %s", self.input_file)
        logging.debug("Expected file: %s", self.expected_file)
        logging.debug("Output file: %s", self.output_file)

    def tearDown(self):
        """
        Clean up files created during testing.
        """
        logging.debug("Tearing down and removing output file if it exists.")
        if os.path.exists(self.output_file):
            logging.debug("Removing file: %s", self.output_file)
            os.remove(self.output_file)

    def test_process_articles(self):
        """
        Test processing articles and verify output against expected JSON.
        """
        logging.debug("Starting `test_process_articles`.")
        print("Output file path:", self.output_file)  # Debug the output path
        # Run the process_articles function
        try:
            process_articles(self.input_file, self.output_file)
            logging.debug("Process completed. Verifying output file exists.")
        except Exception as e:
            logging.error("Error during `process_articles`: %s", e)
            raise

        # Verify the output file exists
        self.assertTrue(os.path.exists(self.output_file), "Output file was not created.")
        logging.debug("Output file successfully created.")

        # Load the output and expected JSON files
        try:
            with open(self.output_file, "r") as f:
                output_data = json.load(f)
            with open(self.expected_file, "r") as f:
                expected_data = json.load(f)
            logging.debug("Loaded output and expected JSON files.")
        except Exception as e:
            logging.error("Error loading JSON files: %s", e)
            raise
        # Print output and expected data for debugging
        print("Generated Output JSON:")
        print(json.dumps(output_data, indent=4))
        print("Expected JSON:")
        print(json.dumps(expected_data, indent=4))

        # Compare the output data with the expected data
        self.assertEqual(
            output_data,
            expected_data,
            "The output JSON does not match the expected JSON."
        )
        logging.debug("Output JSON matches the expected JSON.")

    def test_placeholder(self):
        logging.debug("Running placeholder test.")
        self.assertTrue(True)

    def test_categorize_terms(self):
        """
        Test categorization logic for terms.
        """
        logging.debug("Starting `test_categorize_terms`.")
        term_counts = {
            "PTSD": 2,
            "anxiety": 1,
            "methylation": 3,
            "African-American": 1
        }

        mental_health_categories = ["PTSD", "anxiety", "depression"]
        epigenetic_categories = ["methylation", "BDNF", "FKBP5"]
        ethnographic_categories = {"African descent": ["African-American"]}

        try:
            categorized = categorize_terms(term_counts, {
                "Mental Health": mental_health_categories,
                "Epigenetic": epigenetic_categories,
                "Ethnographic": ethnographic_categories
            })
            logging.debug("Categorization successful: %s", categorized)
        except Exception as e:
            logging.error("Error in `categorize_terms`: %s", e)
            raise

        # Validate categorized counts
        self.assertEqual(categorized["Mental Health"]["PTSD"], 2)
        self.assertEqual(categorized["Epigenetic"]["methylation"], 3)
        self.assertEqual(categorized["Ethnographic"]["African descent"], 1)
        logging.debug("Categorization test passed.")

if __name__ == "__main__":
    logging.debug("Tests are loading...")
    unittest.main()
