import json
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

# Define terms
mental_health_terms = [ 
    "depression", "bipolar", "PTSD", "anxiety", 
    "suicide", "generational trauma",
]
epigenetic_terms = [
    "methylation", "demethylation", "CpG islands", 
    "histone modification", "HPA axis dysregulation", "childhood abuse" 
]
socioeconomic_terms = [
    "low-income", "middle-income", "high-income"
]

# Load preprocessed articles
preprocessed_file = "./preprocessed_articles.json"
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

preprocessed_data = load_json(preprocessed_file)

# Initialize graph
G = nx.Graph()

# Add nodes for each term with category-specific attributes
for term in mental_health_terms:
    G.add_node(term, category="Mental Health")
for term in epigenetic_terms:
    G.add_node(term, category="Epigenetic")
for term in socioeconomic_terms:
    G.add_node(term, category="Socioeconomic")

# Process papers and add edges based on co-occurrence
for paper in preprocessed_data["papers"]:
    term_counts = paper["term_counts"]

    # Combine all terms that appear in the paper
    paper_terms = set()
    for term in mental_health_terms:
        if term_counts["mental health terms"].get(term, 0) > 0:
            paper_terms.add(term)
    for term in epigenetic_terms:
        if term_counts["epigenetic terms"].get(term, 0) > 0:
            paper_terms.add(term)
    for term in socioeconomic_terms:
        if term_counts["socioeconomic terms"].get(term, 0) > 0:
            paper_terms.add(term)

    # Add edges for all pairs of co-occurring terms
    for term1, term2 in combinations(paper_terms, 2):
        if G.has_edge(term1, term2):
            G[term1][term2]["weight"] += 1
        else:
            G.add_edge(term1, term2, weight=1)

# Plot the graph
def plot_graph(G):
    pos = nx.spring_layout(G, seed=42)  # Layout for the graph
    plt.figure(figsize=(12, 12))

    # Node colors based on category
    color_map = {
        "Mental Health": "blue",
        "Epigenetic": "green",
        "Socioeconomic": "orange"
    }
    node_colors = [color_map[G.nodes[node]["category"]] for node in G.nodes]

    # Edge weights for line thickness
    edges, weights = zip(*nx.get_edge_attributes(G, "weight").items())
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w * 0.1 for w in weights])

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=10, label="Mental Health"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=10, label="Epigenetic"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=10, label="Socioeconomic"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.title("Network Graph of Term Co-occurrences")
    plt.axis("off")
    plt.show()

# Call the plot function
plot_graph(G)
