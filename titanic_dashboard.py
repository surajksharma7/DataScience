import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import seaborn as sns
import dash_bootstrap_components as dbc

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Preprocessing
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna('Unknown')
df['deck'] = df['deck'].cat.add_categories('Unknown').fillna('Unknown')  # Proper way for categorical column

# Initialize Dash app with Bootstrap Theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    html.H1("🚢 Titanic Dataset Dashboard", style={'textAlign': 'center', 'marginTop': 20, 'marginBottom': 30}),

    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='pclass-filter',
                options=[{'label': str(cls), 'value': cls} for cls in sorted(df['class'].dropna().unique())],
                placeholder='Select Passenger Class',
                clearable=True
            )
        ], width=6),

        dbc.Col([
            dcc.Dropdown(
                id='sex-filter',
                options=[{'label': gender.capitalize(), 'value': gender} for gender in df['sex'].dropna().unique()],
                placeholder='Select Gender',
                clearable=True
            )
        ], width=6),
    ], className='mb-4'),

    dbc.Row([
        dbc.Col(dcc.Graph(id='survival-by-class'), width=6),
        dbc.Col(dcc.Graph(id='survival-by-sex'), width=6),
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='age-distribution'), width=6),
        dbc.Col(dcc.Graph(id='fare-distribution'), width=6),
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='embarked-distribution'), width=6),
        dbc.Col(dcc.Graph(id='survival-by-deck'), width=6),
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='correlation-heatmap'), width=12),
    ]),

    html.Footer("Dashboard created with ❤️ using Dash and Plotly", style={'textAlign': 'center', 'marginTop': 50})
], fluid=True)

# Callback to update graphs based on filters
@app.callback(
    [
        Output('survival-by-class', 'figure'),
        Output('survival-by-sex', 'figure'),
        Output('age-distribution', 'figure'),
        Output('fare-distribution', 'figure'),
        Output('embarked-distribution', 'figure'),
        Output('survival-by-deck', 'figure'),
        Output('correlation-heatmap', 'figure')
    ],
    [Input('pclass-filter', 'value'),
     Input('sex-filter', 'value')]
)
def update_graphs(selected_class, selected_sex):
    filtered_df = df.copy()

    if selected_class:
        filtered_df = filtered_df[filtered_df['class'] == selected_class]
    if selected_sex:
        filtered_df = filtered_df[filtered_df['sex'] == selected_sex]

    fig1 = px.histogram(filtered_df, x='class', color='survived', barmode='group',
                        title='Survival Count by Class')
    fig2 = px.histogram(filtered_df, x='sex', color='survived', barmode='group',
                        title='Survival Count by Sex')
    fig3 = px.histogram(filtered_df, x='age', nbins=30, color='survived',
                        title='Age Distribution with Survival')
    fig4 = px.box(filtered_df, x='survived', y='fare', color='survived',
                  title='Fare Distribution by Survival')
    fig5 = px.histogram(filtered_df, x='embarked', color='survived', barmode='group',
                        title='Survival by Embarkation Port')
    fig6 = px.histogram(filtered_df, x='deck', color='survived', barmode='group',
                        title='Survival by Deck')
    fig7 = px.imshow(filtered_df.corr(numeric_only=True), text_auto=True,
                     title='Correlation Heatmap')

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7

# Run server
if __name__ == '__main__':
    app.run(debug=True)
    