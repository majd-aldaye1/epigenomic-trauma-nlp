import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from gensim.models.ldamodel import LdaModel
from topic_modeling import build_topic_model
from preprocessing import preprocess_abstracts


# Load Data and Preprocess
def load_and_preprocess_data():
    df = pd.read_csv('data/pubmed_articles.csv')
    df = preprocess_abstracts(df)
    return df

# Function to run LDA and generate topic model
def run_lda(df):
    if isinstance(df['Processed_Abstract'].iloc[0], list):
        all_tokens = df['Processed_Abstract'].tolist()
    else:
        all_tokens = df['Processed_Abstract'].apply(eval).tolist()
    
    lda_model, dictionary, corpus = build_topic_model(all_tokens, num_topics=5, passes=15)
    return lda_model, dictionary, corpus

# Build the interactive 3D topic visualization with Plotly
def create_3d_topic_visualization(lda_model, corpus):
    # Example data for visualization
    topic_x = [1, 2, 3, 4, 5]
    topic_y = [2, 3, 4, 5, 6]
    topic_z = [3, 4, 5, 6, 7]
    topics = ['Epigenetics and Trauma', 'Gene Expression', 'Mental Stress', 'Physical Injury', 'Age-related Changes']
    topic_sizes = [10, 15, 20, 25, 30]  # Adjust sizes dynamically if you wish

    fig = px.scatter_3d(x=topic_x, y=topic_y, z=topic_z,
                        size=topic_sizes, color=topics, hover_name=topics)

    fig.update_layout(
        title='3D Topic Visualization',
        scene=dict(
            xaxis_title='Mental Stress-related Changes',
            yaxis_title='Physical Trauma-related Changes',
            zaxis_title='Age-related Changes'
        )
    )
    return fig

# Dash app setup
app = dash.Dash(__name__)

# Load and preprocess the data
df = load_and_preprocess_data()
lda_model, dictionary, corpus = run_lda(df)

# Layout for the Dash app
app.layout = html.Div([
    html.H1("Epigenetics and Mental Health Topic Exploration"),
    
    # 3D Topic Visualization Graph
    dcc.Graph(
        id='3d_scatter',
        figure=create_3d_topic_visualization(lda_model, corpus)
    ),
    
    # Lambda slider to adjust relevance
    html.Label("Adjust Topic Relevance Metric (λ)"),
    dcc.Slider(
        id='lambda_slider',
        min=0,
        max=1,
        step=0.1,
        value=1,
        marks={0: 'Unique Terms', 1: 'Common Terms'}
    ),
    
    # Dropdown for term mapping (scientific vs mental health terms)
    html.Label("Term Mapping"),
    dcc.Dropdown(
        id='term_mapping',
        options=[
            {'label': 'Scientific Terms', 'value': 'scientific'},
            {'label': 'Mental Health Terms', 'value': 'mental_health'}
        ],
        value='mental_health'  # Default value
    ),

    # Placeholder for interactive updates based on controls
    dcc.Graph(id='updated_3d_graph'),

    # Explanation for relevance metric and saliency
    html.Div([
        html.P("Saliency explains how much a word contributes to a topic based on how frequently it appears and how unique it is to that topic."),
        html.P("Relevance determines how important a word is for a given topic, with λ adjusting the balance between common and unique terms.")
    ])
])

# Define the callbacks for interactivity
@app.callback(
    Output('updated_3d_graph', 'figure'),
    [Input('lambda_slider', 'value'),
     Input('term_mapping', 'value')]
)
def update_graph(lambda_value, term_mapping):
    # Adjust the LDA model with new lambda value (for topic relevance)
    fig = create_3d_topic_visualization(lda_model, corpus)
    fig.update_layout(title=f'Updated with λ = {lambda_value}')
    
    # Logic to replace terms based on term mapping selection (if any logic needed)
    # Placeholder logic: (You can replace with the actual term mapping logic)
    if term_mapping == 'mental_health':
        # Potentially change words in the graph
        pass
    
    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
