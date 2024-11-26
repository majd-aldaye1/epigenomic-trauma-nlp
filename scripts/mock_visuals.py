"""
This script generates mock visualizations for analyzing relationships between socioeconomic, mental health, ethnicity, and epigenetic terms. It uses mock data to simulate insights and present them in an interactive web application.

Key Features:
1. **Heatmaps**:
   - Visualize the correlations between:
     - Socioeconomic and epigenetic terms.
     - Ethnicity and epigenetic terms.
     - Mental health and epigenetic terms.

2. **3D Scatter Plot**:
   - Simplified 3D visualization of the top term associations across mental health, epigenetic, and socioeconomic categories.

3. **Interactive Dashboard**:
   - Combines all visualizations into an aesthetically pleasing and interactive dashboard using Dash.
   - Helps communicate the potential insights and relationships between terms in a visual format.

Applications:
This mock analysis serves as a proof-of-concept for presenting real-world data visualizations, enabling exploration and understanding of how terms in different categories relate to one another.
"""

import json
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from dash import Dash, html, dcc
import plotly.graph_objects as go

# Define mock counts for visualization
mock_counts = defaultdict(lambda: defaultdict(int))

# Mock data simulating insightful trends
# Mental health counts
mock_counts["mental_health"] = {
    "depression": 50,
    "bipolar": 30,
    "PTSD": 70,
    "anxiety": 40,
    "suicide": 20,
    "generational trauma": 60
}

# Epigenetic counts
mock_counts["epigenetic"] = {
    "methylation": 80,
    "demethylation": 60,
    "CpG islands": 50,
    "histone modification": 90,
    "HPA axis dysregulation": 70,
    "childhood abuse": 30
}

# Socioeconomic counts
mock_counts["socioeconomic"] = {
    "low-income": 100,
    "middle-income": 50,
    "high-income": 20
}

# Ethnicity counts
mock_counts["ethnicity"] = {
    "african descent": 80,
    "latino/hispanic descent": 60,
    "asian descent": 40,
    "indigenous descent": 90,
    "arab descent": 30,
    "european descent": 20
}

# Function to prepare heatmap data
def prepare_heatmap_data(category1, category2, counts):
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
    plt.title(title)
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

# Generate heatmaps using mock data
df1 = prepare_heatmap_data("socioeconomic", "epigenetic", mock_counts)
heatmap1 = create_heatmap(df1, "socioeconomic", "epigenetic", "Socioeconomic vs. Epigenetic Terms (Mock)")

df2 = prepare_heatmap_data("ethnicity", "epigenetic", mock_counts)
heatmap2 = create_heatmap(df2, "ethnicity", "epigenetic", "Ethnicity vs. Epigenetic Terms (Mock)")

df3 = prepare_heatmap_data("mental_health", "epigenetic", mock_counts)
heatmap3 = create_heatmap(df3, "mental_health", "epigenetic", "Mental Health vs. Epigenetic Terms (Mock)")

# Simplified 3D Scatter Plot
def create_simple_3d_graph(counts):
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
        title="Simplified 3D Scatter Plot: Top Terms (Mock)",
        scene=dict(
            xaxis_title="Top Mental Health Terms",
            yaxis_title="Top Epigenetic Terms",
            zaxis_title="Top Socioeconomic Terms"
        )
    )
    return fig

mock_graph_3d = create_simple_3d_graph(mock_counts)

# Create Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Mock Term Analysis and Relationships", style={"textAlign": "center"}),

    html.Div([
        html.H2("Socioeconomic vs. Epigenetic Terms (Mock)"),
        html.Img(src=heatmap1, style={"width": "100%", "height": "auto"})
    ], style={"marginBottom": "50px"}),

    html.Div([
        html.H2("Ethnicity vs. Epigenetic Terms (Mock)"),
        html.Img(src=heatmap2, style={"width": "100%", "height": "auto"})
    ], style={"marginBottom": "50px"}),

    html.Div([
        html.H2("Mental Health vs. Epigenetic Terms (Mock)"),
        html.Img(src=heatmap3, style={"width": "100%", "height": "auto"})
    ], style={"marginBottom": "50px"}),

    html.Div([
        html.H2("Simplified 3D Scatter Plot: Top Terms (Mock)"),
        dcc.Graph(figure=mock_graph_3d)
    ])
])

if __name__ == "__main__":
    app.run_server(debug=True, port=8051)
