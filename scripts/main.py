import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Load Data
def load_data():
    try:
        df = pd.read_csv('data/preprocessed_pubmed_articles.csv')
        print(df.columns)
        return df
    except FileNotFoundError:
        print("Error: The file 'preprocessed_pubmed_articles.csv' was not found.")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

# Build the interactive 3D topic visualization with Plotly
def create_3d_topic_visualization(df, topic_x='Mental_Health_Terms', topic_y='Epigenetic_Terms', topic_z='Socioeconomic_Terms'):
    if df.empty:
        return go.Figure()  # Return an empty figure if data is not loaded properly

    fig = px.scatter_3d(df,
                        x=topic_x,
                        y=topic_y,
                        z=topic_z,
                        color='Ethnographic_Terms',  # Color based on race/ethnicity
                        hover_name='Journal',  # Display journal or other relevant info on hover
                        title='3D Topic Visualization: Epigenetics, Mental Health, Socioeconomic Factors')

    fig.update_layout(
        scene=dict(
            xaxis_title='Mental Health Terms',
            yaxis_title='Epigenetic Changes',
            zaxis_title='Socioeconomic Disparities'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

# Dash app setup
app = dash.Dash(__name__)

# Load the preprocessed data
df = load_data()

# Layout for the Dash app
app.layout = html.Div([
    html.H1("Epigenetics and Mental Health Correlation Exploration"),

    # 3D Topic Visualization Graph
    dcc.Graph(
        id='3d_scatter',
        figure=create_3d_topic_visualization(df)
    ),
    
    # Dropdown for choosing which axis to visualize
    html.Label("Select X-axis (Mental Health Terms)"),
    dcc.Dropdown(
        id='xaxis_dropdown',
        options=[{'label': term, 'value': term} for term in df['Mental_Health_Terms'].unique()],
        value='Mental_Health_Terms'  # Default value
    ),

    html.Label("Select Y-axis (Epigenetic Terms)"),
    dcc.Dropdown(
        id='yaxis_dropdown',
        options=[{'label': term, 'value': term} for term in df['Epigenetic_Terms'].unique()],
        value='Epigenetic_Terms'  # Default value
    ),

    html.Label("Select Z-axis (Socioeconomic Terms)"),
    dcc.Dropdown(
        id='zaxis_dropdown',
        options=[{'label': term, 'value': term} for term in df['Socioeconomic_Terms'].unique()],
        value='Socioeconomic_Terms'  # Default value
    )
])

# Define the callback for interactivity
@app.callback(
    Output('3d_scatter', 'figure'),
    [Input('xaxis_dropdown', 'value'),
     Input('yaxis_dropdown', 'value'),
     Input('zaxis_dropdown', 'value')]
)
def update_graph(xaxis, yaxis, zaxis):
    return create_3d_topic_visualization(df, topic_x=xaxis, topic_y=yaxis, topic_z=zaxis)

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
