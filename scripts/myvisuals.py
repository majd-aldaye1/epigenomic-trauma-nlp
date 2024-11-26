import json
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from dash import Dash, html, dcc
import plotly.graph_objects as go

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
ethnicity_terms = [
    "african descent", "latino/hispanic descent", "asian descent",
    "indigenous descent", "arab descent", "european descent"
]

# File paths
preprocessed_file = "./preprocessed_articles.json"

# Load JSON data
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

preprocessed_data = load_json(preprocessed_file)

# Initialize counts dictionary
counts = defaultdict(lambda: defaultdict(int))

# Aggregate counts for each term
for paper in preprocessed_data["papers"]:
    term_counts = paper["term_counts"]
    for term in mental_health_terms:
        counts["mental_health"][term] += term_counts["mental health terms"].get(term, 0)
    for term in epigenetic_terms:
        counts["epigenetic"][term] += term_counts["epigenetic terms"].get(term, 0)
    for term in socioeconomic_terms:
        counts["socioeconomic"][term] += term_counts["socioeconomic terms"].get(term, 0)
    for term in ethnicity_terms:
        counts["ethnicity"][term] += term_counts["ethnographic terms"].get(term, 0)

# Function to prepare heatmap data
def prepare_heatmap_data(category1, category2):
    data = []
    for term1 in counts[category1]:
        for term2 in counts[category2]:
            combined_count = counts[category1][term1] + counts[category2][term2]
            data.append({
                category1: term1,
                category2: term2,
                "Count": combined_count
            })
    return pd.DataFrame(data)

# Function to create heatmap and save as base64 image
def create_heatmap(df, category1, category2, title):
    heatmap_data = df.pivot_table(
        index=category1,
        columns=category2,
        values="Count",
        aggfunc="sum",
        fill_value=0
    )
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
    # Focus on the top 3 terms in each category for simplicity
    top_mh_terms = sorted(counts["mental_health"].items(), key=lambda x: -x[1])[:3]
    top_ep_terms = sorted(counts["epigenetic"].items(), key=lambda x: -x[1])[:3]
    top_socio_terms = sorted(counts["socioeconomic"].items(), key=lambda x: -x[1])[:3]

    # Prepare data for the simplified plot
    x, y, z, sizes = [], [], [], []
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
