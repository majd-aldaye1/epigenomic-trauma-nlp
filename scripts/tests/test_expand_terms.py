import unittest
from unittest.mock import patch, MagicMock
from scripts.expand_terms import (
    is_valid_title,
    fetch_wikipedia_corpus,
    generate_similar_terms,
    expand_and_save_to_json
)

import json
import numpy as np

class TestExpandTerms(unittest.TestCase):
    """
    Unit tests for expand_terms.py functionality.
    """

    def test_is_valid_title(self):
        """
        Test filtering of valid and invalid Wikipedia titles.
        """
        self.assertTrue(is_valid_title("Mental health"))
        self.assertTrue(is_valid_title("DNA methylation"))
        self.assertFalse(is_valid_title("Help:Page"))
        self.assertFalse(is_valid_title("File:Example"))
        self.assertFalse(is_valid_title("Invalid@Title"))

    @patch("scripts.expand_terms.wiki_wiki.page")
    def test_fetch_wikipedia_corpus(self, mock_page):
        """
        Test fetching Wikipedia corpus based on input terms.
        """
        mock_page.return_value = MagicMock(
            exists=MagicMock(return_value=True),
            links={"Link1": MagicMock(), "Link2": MagicMock()},
            title="Mock Page"
        )

        result = fetch_wikipedia_corpus(["Mental health"], max_depth=1, max_pages=5)
        self.assertIn("Mock Page", result)
        self.assertLessEqual(len(result), 5)

    @patch("scripts.expand_terms.util.cos_sim")
    def test_generate_similar_terms(self, mock_cos_sim):
        """
        Test generation of similar terms using mocked embeddings.
        """
        # Mock `model.encode` with fixed-length embeddings
        mock_encode = lambda term: np.array([ord(c) for c in term[:5]] + [0] * (5 - len(term[:5])))  # Fixed length of 5

        # Mock similarity scores
        mock_cos_sim.side_effect = lambda x, y: np.dot(x, y.T)  # Simulate cosine similarity

        # Define the mock input
        corpus = ["place", "tea", "stress", "tree", "ear", "child", "arab", "hospital", "see"]
        term_list = ["mental health"]

        # Call the function with the mock encoder
        result = generate_similar_terms(term_list, model=None, corpus=corpus, topn=2, per_term=True, mock_encode=mock_encode)

        # Debugging: Print the result
        print("Result:", result)

        # Assertions
        self.assertIn("mental health", result)
        self.assertIn("stress", result["mental health"]) # Mocked result should include "stress"
        self.assertIn("mental health", result["mental health"]) # Mocked result should include "mental health" (the term itslef)
        self.assertEqual(len(result["mental health"]), 2)  # Top 2 terms

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("scripts.expand_terms.expand_terms_for_query")
    def test_expand_and_save_to_json(self, mock_expand_terms_for_query, mock_open):
        """
        Test the entire pipeline of expanding terms and saving to a JSON file.
        """
        # Mock the output of expand_terms_for_query
        mock_expand_terms_for_query.side_effect = lambda term, **kwargs: [term, "related_term1", "related_term2"]

        # Call the function
        expand_and_save_to_json(
            mental_health_terms=["mental health"],
            epigenetic_terms=["DNA methylation"],
            ethnographic_terms={"African descent": ["Black person"]},
            socioeconomic_terms=["low-income"]
        )

        # Retrieve the written data from the mock file handle
        handle = mock_open()
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)

        # Parse the written JSON
        data = json.loads(written_data)

        # Assertions to validate the structure and content of the JSON
        self.assertIn("mental health terms", data)
        self.assertIn("mental health", data["mental health terms"])
        self.assertEqual(data["mental health terms"], ["related_term1", "related_term2"])

        self.assertIn("epigenetic terms", data)
        self.assertIn("dna methylation", data["epigenetic terms"])
        self.assertEqual(data["epigenetic terms"], ["related_term1", "related_term2"])

        self.assertIn("socioeconomic terms", data)
        self.assertIn("low-income", data["socioeconomic terms"])
        self.assertEqual(data["socioeconomic terms"], ["related_term1", "related_term2"])

        self.assertIn("ethnographic terms", data)
        self.assertIn("african descent", data["ethnographic terms"])
        self.assertEqual(data["ethnographic terms"], ["black person"])


if __name__ == "__main__":
        unittest.main()
        
