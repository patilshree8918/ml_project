
# Importing the libraries
import pickle
import pandas as pd
import webbrowser
import dash
# from dash_core_components import dcc
from dash import dcc
# import dash_html_components as html
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go   
import numpy as np

data = pd.read_csv(r"balanced_reviews.csv")

data.isnull().sum() 

data[data.isnull().any(axis=1)] 
data.dropna(inplace=True) 
data = data[data['overall']!=3] 
data['Positivity'] = np.where(data['overall'] > 3, 1, 0) 

labels = ["Positive","Negative","Neutral"]

values=[len(data[data.Positivity == 1]), len(data[data.Positivity == 0])]
scrape = pd.read_csv(r"scrappedReviews.csv")
scrape.head()
 
graph = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
 

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
project_name = "Sentiment Analysis with Insights"

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

def load_model():
    global pickle_model
    global vocab
    global scrappedReviews
    
    scrappedReviews = pd.read_csv('scrappedReviews.csv')
    
    file = open("pickle_model.pkl", 'rb')
    pickle_model = pickle.load(file)

    file = open("feature.pkl", 'rb') 
    vocab = pickle.load(file)
        
def check_review(reviewText):

    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    reviewText = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    return pickle_model.predict(reviewText)

def create_app_ui():

    global project_name
    main_layout = dbc.Container(
        dbc.Jumbotron(
                [
                    
                    html.H1(id = 'heading', children = project_name, className = 'display-3 mb-4'),

                    html.Div([html.H1(children='Pie Chart', id='pieh1',style={'padding':5})]),
                            dcc.Graph(figure = graph, style={'width': '100%','height':700,},className = 'd-flex justify-content-center'),

                    html.Br(),html.Hr(),html.Br(),

                    html.Div([html.H1(children='Enter Review', id='dropdown_h1',style={'padding':5})],style={'backgroundColor':'#bdb9b9','height': 60}),
                    html.Br(),html.Br(),

                    dbc.Textarea(id = 'textarea', className="mb-3", placeholder="Enter the Review", value = '', style = {'height': '150px'}),
                     html.Div([html.H1(children='Choose Review', id='dropdown_h2',style={'padding':5})],style={'backgroundColor':'#bdb9b9','height': 60}),
                           html.Br(),html.Br(),
                    dbc.Container([
                        dcc.Dropdown(
                    id='dropdown',
                    placeholder = 'Select a Review',
                    options=[{'label': i[:100] + "...", 'value': i} for i in scrappedReviews.reviews],
                    value = '',
                    style = {'margin-bottom': '30px'}
                    
                )
                       ],
                        style = {'padding-left': '50px', 'padding-right': '50px'}
                        ),
                    dbc.Button("Submit", color="dark", className="mt-2 mb-3", id = 'button', style = {'width': '100px'}),
                    html.Div(id = 'result'),
                    html.Div(id = 'result1')
                    ],
                className = 'text-center'
                ),
        className = 'mt-4'
        )
    
    return main_layout

@app.callback(
    Output('result', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
    State('textarea', 'value')
    ]
    )
      
def update_app_ui(n_clicks, textarea):

    if(textarea == ""):
        return dbc.Alert("Neutral", color="dark")

    result_list = check_review(textarea)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")

@app.callback(
    Output('result1', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
     State('dropdown', 'value')
     ]
    )

def update_dropdown(n_clicks, value):

    if(value == ''):
        return dbc.Alert("Neutral", color="dark")

    result_list = check_review(value)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")
    
def main():
    
    global app
    global project_name
    load_model()
    open_browser()
    app.layout = create_app_ui()
    app.title = project_name
    app.run_server()
    app = None
    project_name = None

if __name__ == '__main__':
    main()


