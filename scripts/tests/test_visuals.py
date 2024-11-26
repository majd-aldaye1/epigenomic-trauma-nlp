import plotly.graph_objects as go  # Import go for assertions
from collections import defaultdict
import unittest
from scripts.myvisuals import prepare_heatmap_data, create_simple_3d_graph
from collections import defaultdict

class TestVisuals(unittest.TestCase):

    def setUp(self):
        # Toy data for testing
        self.toy_counts = defaultdict(lambda: defaultdict(int))
        self.toy_counts["mental_health"]["depression"] = 10
        self.toy_counts["mental_health"]["anxiety"] = 5
        self.toy_counts["epigenetic"]["methylation"] = 7
        self.toy_counts["epigenetic"]["CpG islands"] = 3
        self.toy_counts["socioeconomic"]["low-income"] = 8
        self.toy_counts["socioeconomic"]["middle-income"] = 4

    def test_prepare_heatmap_data(self):
        df = prepare_heatmap_data("mental_health", "epigenetic")
        self.assertFalse(df.empty)
        self.assertIn("mental_health", df.columns)
        self.assertIn("epigenetic", df.columns)

    def test_create_simple_3d_graph(self):
        graph = create_simple_3d_graph()
        self.assertIsInstance(graph, go.Figure)

if __name__ == "__main__":
    unittest.main()
