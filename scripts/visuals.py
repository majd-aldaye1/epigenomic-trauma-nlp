import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Load your preprocessed data
def load_preprocessed_data(filepath):
    df = pd.read_csv(filepath)
    
    # Handle NaN values by filling them with 'Unknown'
    df['Ethnographic_Terms'] = df['Ethnographic_Terms'].fillna('Unknown')
    df['Socioeconomic_Terms'] = df['Socioeconomic_Terms'].fillna('Unknown')
    df['Mental_Health_Terms'] = df['Mental_Health_Terms'].fillna('Unknown')
    df['Epigenetic_Terms'] = df['Epigenetic_Terms'].fillna('Unknown')
    
    return df

# Load the dataset
df = load_preprocessed_data('data/preprocessed_pdf_content.csv')

# Initialize Dash app
app = dash.Dash(__name__)

# Layout for Dash App (HTML interface)
app.layout = html.Div([
    html.H1("Advanced Interactive Visualization Dashboard"),

    # Slider for dynamic interaction across all graphs
    html.Div([
        html.Label("Adjust visualization parameters using the slider:"),
        dcc.Slider(
            id='slider', 
            min=1, 
            max=10, 
            step=1, 
            value=5, 
            marks={i: f'{i}' for i in range(1, 11)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'width': '50%', 'margin': 'auto', 'padding': '20px'}),

    # Create a 2-row grid with 3 columns per row for all six visualizations
    html.Div([

        # Row 1: Visualizations 1, 2, and 3
        html.Div([
            html.Div([dcc.Graph(id='graph1')], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([dcc.Graph(id='graph2')], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([dcc.Graph(id='graph3')], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),
        ], style={'display': 'flex', 'justify-content': 'space-between'}),

        # Row 2: Visualizations 4, 5, and 6
        html.Div([
            html.Div([dcc.Graph(id='graph4')], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([dcc.Graph(id='graph5')], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([dcc.Graph(id='graph6')], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),
        ], style={'display': 'flex', 'justify-content': 'space-between'})

    ], style={'padding': '20px'}),
])

# Callback to update all graphs based on the slider value
@app.callback(
    [Output('graph1', 'figure'),
     Output('graph2', 'figure'),
     Output('graph3', 'figure'),
     Output('graph4', 'figure'),
     Output('graph5', 'figure'),
     Output('graph6', 'figure')],
    [Input('slider', 'value')]
)
def update_graphs(slider_value):
    # Visualization 1: Treemap for Mental Health Across Racial & Socioeconomic Factors
    fig1 = px.treemap(df, path=['Ethnographic_Terms', 'Socioeconomic_Terms', 'Mental_Health_Terms'], 
                      title="Mental Health Disorders Across Racial & Socioeconomic Factors")

    # Visualization 2: 3D Scatter Plot for Epigenetics, Mental Health, and Socioeconomic Factors
    fig2 = px.scatter_3d(df, x='Epigenetic_Terms', y='Mental_Health_Terms', z='Socioeconomic_Terms', 
                         color='Mental_Health_Terms', title="Epigenetics vs Mental Health and Socioeconomic Factors")
    fig2.update_traces(marker=dict(size=slider_value*2))

    # Visualization 3: Heatmap for Mental Health and Socioeconomic Factors
    pivot_table = df.pivot_table(index='Mental_Health_Terms', columns='Socioeconomic_Terms', aggfunc='size', fill_value=0)
    fig3 = px.imshow(pivot_table, title="Heatmap: Mental Health vs Socioeconomic Factors")

    # Visualization 4: Sunburst Chart for Risk of Suicide and Epigenetic Impacts
    fig4 = px.sunburst(df, path=['Epigenetic_Terms', 'Ethnographic_Terms', 'Socioeconomic_Terms', 'Mental_Health_Terms'], 
                       title="Risk of Suicide and Epigenetic Impacts")

    # Visualization 5: 3D Scatter Plot for Mental Health, Race, and Socioeconomic Factors
    fig5 = px.scatter_3d(df, x='Ethnographic_Terms', y='Socioeconomic_Terms', z='Mental_Health_Terms', 
                         color='Ethnographic_Terms', title="Mental Health Across Racial and Socioeconomic Factors")
    fig5.update_traces(marker=dict(size=slider_value*2))

    # Visualization 6: Creative Network Visualization (Force-directed Network Graph)
    fig6 = go.Figure(go.Scatter(
        x=[1, 2, 3, 4], y=[1, 3, 2, 4], text=["PTSD", "DNA Methylation", "Income Inequality", "Race"],
        mode="markers+text", textposition="top center", marker=dict(size=[slider_value*10, 20, 15, 10])))
    fig6.update_layout(title="Creative Network Visualization", showlegend=False)

    return fig1, fig2, fig3, fig4, fig5, fig6

# Run Dash App
if __name__ == '__main__':
    app.run_server(debug=True)
