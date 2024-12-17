"""
This script creates a set of interactive and static visualizations to analyze relationships between socioeconomic, mental health, and epigenetic terms using data preprocessed from biomedical literature.

Key Features:
1. **Data Aggregation**:
   - Extracts term frequencies for mental health, epigenetics, socioeconomic, and ethnicity categories from a preprocessed JSON file.

2. **Heatmaps**:
   - Generates heatmaps visualizing correlations between:
     - Socioeconomic and epigenetic terms.
     - Ethnicity and epigenetic terms.
     - Mental health and epigenetic terms.

3. **3D Scatter Plot**:
   - Creates a simplified 3D scatter plot to visualize the strongest term associations across mental health, epigenetic, and socioeconomic categories.

4. **Interactive Dashboard**:
   - Combines all visualizations into an interactive web dashboard using Dash.
   - Provides insights into the epigenomic impact of social trauma across different dimensions.

Applications:
This tool supports the meta-analysis project by offering a visual narrative of term relationships, making it easier to identify patterns and insights in the data.
"""

import json, base64, logging
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from dash import Dash, html, dcc
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log format
    handlers=[logging.StreamHandler()]  # Ensure logs are sent to the console
)

# File paths
processed_file = "./final_modeling.json"  # Output file from modeling.py

# Load processed topics data
def load_json(file_path):
    """
    Load JSON data from a file and validate its structure.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        if not isinstance(data, dict) or "papers" not in data:
            raise ValueError("Invalid JSON structure: Expected a dictionary with a 'papers' key.")
        if not isinstance(data["papers"], list):
            raise ValueError("Invalid JSON structure: 'papers' key must contain a list of dictionaries.")
        return data["papers"]  # Return the list of papers

# Correct JSON loading
processed_data = load_json(processed_file)
# Debugging: Print the structure of processed_data
logging.info(f"Loaded processed data: {processed_data}")

# Define terms
mental_health_terms = [ 
    "depression", "bipolar", "PTSD", "anxiety", 
    "suicide", "generational trauma",
]
epigenetic_terms = [
    "methylation", "CpG islands", 
    "histone modification", "HPA axis dysregulation", "FKBP5", "BDNF"
]
socioeconomic_terms = [
    "low-income", "middle-income", "high-income"
]
ethnicity_terms = [
    "african descent", "latino/hispanic descent", "asian descent",
    "indigenous descent", "arab descent", "european descent"
]

# Function to flatten terms into pre-defined categories
def flatten_to_predefined_categories(papers):
    """
    Map terms in categorized_counts to pre-defined categories.
    Terms not matching the pre-defined categories will be skipped.
    """
    counts = defaultdict(lambda: defaultdict(int))
    logging.info(f"Flattening categorized counts for {len(papers)} papers...")

    for paper_idx, paper in enumerate(papers, start=1):
        logging.info(f"Processing paper ID: {paper.get('paper_id', 'Unknown')}, Index: {paper_idx}")

        # Validate the structure of the paper
        if "categorized_counts" not in paper:
            logging.warning(f"Skipping paper {paper_idx} without 'categorized_counts'.")
            continue

        categorized = paper["categorized_counts"]
        # Aggregate counts for mental health terms
        if "Mental Health" in categorized:
            for term, count in categorized["Mental Health"].items():
                #if term.lower() in [t.lower() for t in mental_health_terms]:
                    counts["mental_health"][term] += count
                    logging.debug(f"Added {count} to 'mental_health' category for term '{term}'.")

        # Aggregate counts for epigenetic terms
        if "Epigenetic" in categorized:
            for term, count in categorized["Epigenetic"].items():
                #if term.lower() in [t.lower() for t in epigenetic_terms]:
                    counts["epigenetic"][term] += count
                    logging.debug(f"Added {count} to 'epigenetic' category for term '{term}'.")
                # else:
                #     logging.warning(f"Term '{term}' in 'Epigenetic' does not match predefined terms and is skipped.")

        # Aggregate counts for socioeconomic terms
        if "Socioeconomic" in categorized:
            for term, count in categorized["Socioeconomic"].items():
                #if term.lower() in [t.lower() for t in socioeconomic_terms]:
                    counts["socioeconomic"][term] += count
                    logging.debug(f"Added {count} to 'socioeconomic' category for term '{term}'.")

        # Aggregate counts for ethnicity terms
        if "Ethnographic" in categorized:
            for term, count in categorized["Ethnographic"].items():
                #if term.lower() in [t.lower() for t in ethnicity_terms]:
                    counts["ethnicity"][term] += count
                    logging.debug(f"Added {count} to 'ethnicity' category for term '{term}'.")
                # else:
                #     logging.warning(f"Term '{term}' in 'Ethnographic' does not match predefined terms and is skipped.")
        logging.debug(f"Categorized counts for paper {paper_idx}: {categorized}")
        logging.debug(f"Counts after processing paper {paper_idx}: {counts}")

    logging.info("Flattening complete. Aggregated counts ready.")
    logging.info(f"Flattened counts: {counts}")
    return counts

# Initialize counts dictionary
counts = flatten_to_predefined_categories(processed_data)
logging.info(f"Flattened counts after processing all papers: {dict(counts)}")

# Function to prepare heatmap data
def prepare_heatmap_data(category1, category2):
    """
    Prepare heatmap data by aggregating counts for two categories.
    """
    data = []
    logging.info(f"Preparing heatmap data for {category1} vs {category2}...")
    for term1 in counts[category1]:
        for term2 in counts[category2]:
            logging.debug(f"Combining terms: {term1} ({counts[category1][term1]}) and {term2} ({counts[category2][term2]})")
            combined_count = counts[category1][term1] + counts[category2][term2]
            data.append({
                category1: term1,
                category2: term2,
                "Count": combined_count
            })
            logging.debug(f"Current row: {{'{category1}': '{term1}', '{category2}': '{term2}', 'Count': {combined_count}}}")
            logging.debug(f"Aggregating {term1} ({counts[category1][term1]}) and {term2} ({counts[category2][term2]})")
    logging.info(f"Prepared heatmap data for {category1} vs {category2}:\n{data}")
    return pd.DataFrame(data)

# Function to create heatmap and save as base64 image
def create_heatmap(df, category1, category2, title):
    """
    Create a heatmap visualization from a dataframe.
    """
    if df.empty:
        logging.warning(f"No data to generate heatmap for {category1} vs {category2}.")
        return None
    logging.info(f"DataFrame content:\n{df}")

    logging.debug(f"DataFrame before pivot for {category1} vs {category2}: {df}")
    try:
        heatmap_data = df.pivot_table(
            index=category1,
            columns=category2,
            values="Count",
            aggfunc="sum",
            fill_value=0
        )
        logging.info(f"Pivot table created successfully for {category1} vs {category2}.")
    except KeyError as e:
        logging.error(f"Missing expected column: {e}")
        raise

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_data,
        cmap="coolwarm",
        annot=True,
        fmt="d",
        linewidths=0.5,
        xticklabels=True,
        yticklabels=True
    )
    plt.title(title, fontsize=14)
    plt.xlabel(f"{category2.capitalize()} Terms")
    plt.ylabel(f"{category1.capitalize()} Terms")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save as base64 image
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# Generate heatmaps
df1 = prepare_heatmap_data("socioeconomic", "epigenetic")
heatmap1 = create_heatmap(df1, "socioeconomic", "epigenetic", "Socioeconomic vs. Epigenetic Terms")

df2 = prepare_heatmap_data("ethnicity", "epigenetic")
heatmap2 = create_heatmap(df2, "ethnicity", "epigenetic", "Ethnicity vs. Epigenetic Terms")

df3 = prepare_heatmap_data("mental_health", "epigenetic")
heatmap3 = create_heatmap(df3, "mental_health", "epigenetic", "Mental Health vs. Epigenetic Terms")

# Simplified 3D Scatter Plot
def create_simple_3d_graph():
    """
    Create a simplified 3D scatter plot with top terms in each category.
    """
    # Focus on the top 3 terms in each category for simplicity
    top_mh_terms = sorted(counts["mental_health"].items(), key=lambda x: -x[1])[:3]
    top_ep_terms = sorted(counts["epigenetic"].items(), key=lambda x: -x[1])[:3]
    top_socio_terms = sorted(counts["socioeconomic"].items(), key=lambda x: -x[1])[:3]

    # Prepare data for the simplified plot
    x, y, z, sizes = [], [], [], []
    logging.debug(f"3D Scatter Data: x={x}, y={y}, z={z}, sizes={sizes}")
    for mh_term, mh_count in top_mh_terms:
        for ep_term, ep_count in top_ep_terms:
            for socio_term, socio_count in top_socio_terms:
                combined_count = mh_count + ep_count + socio_count
                x.append(mh_term)
                y.append(ep_term)
                z.append(socio_term)
                sizes.append(combined_count)

    # Create the simplified 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=sizes,
                color=sizes,
                colorscale='Viridis',
                opacity=0.8
            ),
            text=[f"{mh}, {ep}, {socio}" for mh, ep, socio in zip(x, y, z)]
        )
    ])
    fig.update_layout(
        title="Simplified 3D Scatter Plot: Top Terms",
        scene=dict(
            xaxis_title="Top Mental Health Terms",
            yaxis_title="Top Epigenetic Terms",
            zaxis_title="Top Socioeconomic Terms"
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    return fig

simple_graph_3d = create_simple_3d_graph()

# Create Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Epigenomic Impact of Social Trauma: A Visual Analysis", style={"textAlign": "center", "fontFamily": "Arial", "color": "#333"}),

    html.H2("By Epigenomic NLP | CS 4701, Cornell University", style={"textAlign": "center", "fontStyle": "italic", "color": "#666"}),

    html.Div([html.H3("Socioeconomic vs. Epigenetic Terms"), html.Img(src=heatmap1)], style={"marginBottom": "50px"}),

    html.Div([html.H3("Ethnicity vs. Epigenetic Terms"), html.Img(src=heatmap2)], style={"marginBottom": "50px"}),

    html.Div([html.H3("Mental Health vs. Epigenetic Terms"), html.Img(src=heatmap3)], style={"marginBottom": "50px"}),

    html.Div([html.H3("3D Visualization of Term Associations"), dcc.Graph(figure=simple_graph_3d)])
])

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)