import numpy as np
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import mlflow
import os

app = dash.Dash(__name__)
app.title = "Car Prediction App"


# Add any initialization code here if needed
mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th/")
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"
os.environ["LOGNAME"] = "st124963-a3"

# model_name = "st12-a3-model"
model_name = "st124963-a3-model"
model_version = 1

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

app.layout = html.Div([
    html.H1("Car Selling Price Prediction", style={'text-align': 'center', 'color': '#4a4a4a', 'margin-bottom': '20px'}),
    
    # Instructions Section
    html.Div([
        html.H3("Instructions:", style={'color': '#2c3e50', 'margin-bottom': '10px'}),
        html.Ul([
            html.Li("Enter the year of manufacture for the car."),
            html.Li("Specify the engine capacity in CC."),
            html.Li("Provide the total kilometers driven."),
            html.Li("Enter the car's mileage in km/l."),
            html.Li("If nothing is provided, default values will be used."),
            html.Li("Using the model, it will predict if the car is cheap, moderate, expensive or very expensive"),
        ], style={'margin-left': '20px', 'color': '#34495e', 'font-size': '16px'}),
    ], style={'padding': '15px', 'border': '1px solid #bdc3c7', 'border-radius': '5px', 'background-color': '#ecf0f1', 'margin-bottom': '20px'}),
    
    # Form Section
    html.Div([
        # Input Field: Year
        html.Div([
            html.Label("Year:", style={'font-weight': 'bold', 'color': '#2c3e50'}),
            dcc.Input(id='input-1', type='text', placeholder='Enter year...', style={ 'padding': '10px', 'margin-bottom': '10px', 'border': '1px solid #bdc3c7', 'border-radius': '5px'}),
        ], style={'display': 'flex', 'flex-direction': 'column'}),
        
        # Input Field: Engine
        html.Div([
            html.Label("Engine:", style={'font-weight': 'bold', 'color': '#2c3e50'}),
            dcc.Input(id='input-2', type='text', placeholder='Enter engine capacity...', style={ 'padding': '10px', 'margin-bottom': '10px', 'border': '1px solid #bdc3c7', 'border-radius': '5px'}),
        ], style={'display': 'flex', 'flex-direction': 'column'}),
        
        # Input Field: Kilometer Driven
        html.Div([
            html.Label("Kilometer Driven:", style={'font-weight': 'bold', 'color': '#2c3e50'}),
            dcc.Input(id='input-3', type='text', placeholder='Enter kilometers driven...', style={ 'padding': '10px', 'margin-bottom': '10px', 'border': '1px solid #bdc3c7', 'border-radius': '5px'}),
        ], style={'display': 'flex', 'flex-direction': 'column'}),
        
        # Input Field: Mileage
        html.Div([
            html.Label("Mileage:", style={'font-weight': 'bold', 'color': '#2c3e50'}),
            dcc.Input(id='input-4', type='text', placeholder='Enter mileage...', style={ 'padding': '10px', 'margin-bottom': '20px', 'border': '1px solid #bdc3c7', 'border-radius': '5px'}),
        ], style={'display': 'flex', 'flex-direction': 'column'}),

    ], style={'display': 'grid', 
        'grid-template-columns': '1fr 1fr', 
        'gap': '20px',
        'margin-bottom': '20px'}),
    
    # Submit Button
    html.Button('Submit', id='submit-button', n_clicks=0, style={'background-color': '#3498db', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
    
    # Output Div
    # html.Div(id='output-div', style={'margin-top': '20px', 'padding': '15px', 'border': '1px solid #bdc3c7', 'border-radius': '5px', 'background-color': '#f8f9fa'})
    dcc.Markdown(id='output-div', style={
        'margin-top': '20px', 
        'padding': '15px', 
        'border': '1px solid #bdc3c7', 
        'border-radius': '5px', 
        'background-color': '#f8f9fa'
    })
], style={'width': '50%', 'margin': '0 auto', 'padding': '20px', 'border': '1px solid #bdc3c7', 'border-radius': '10px', 'background-color': '#f7f7f7'})

# prediction function
def prediction(year: float, engine: float, km_driven: float, mileage: float) -> np.ndarray:
    """
    Predict the car's selling price based on input features.

    :param year: The year of the car.
    :param engine: The engine power of the car.
    :param km_driven: The total kilometers driven by the car.
    :param mileage: The mileage of the car.
    :return: The predicted category of selling price of the car.
    """
    try:
        features = {
            'year': [year],
            'engine': [engine],
            'km_driven': [km_driven],
            'mileage':[mileage]
        }

        X = pd.DataFrame(features, index=[0])
        X = X.to_numpy()
        prediction = model.predict(X)
        
        return prediction
    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")
        

def getDefaultValue() -> tuple:
    """
    Get the default values for the input fields.
    """

    df = pd.read_csv('data/Cars.csv')
    owner_map = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4,
        'Test Drive Car': 5
    }

    df['owner'] = df['owner'].map(owner_map)
    df.drop(df[(df['fuel'] == 'LPG') | (df['fuel'] == 'CNG')].index, inplace = True)
    df['mileage'] = df['mileage'].str.replace('kmpl', '')
    df['mileage'] = df['mileage'].astype(float)
    df['engine'] = df['engine'].str.replace('CC', '')
    df['engine'] = df['engine'].astype(float)
    df.drop('torque', axis = 1, inplace = True)
    df.drop(df[df['owner'] == 5].index, inplace = True)

    median_year = df['year'].median() # since the data is right skewed, median is a better measure of central tendency
    median_km_driven = df['km_driven'].median()
    median_engine = df['engine'].median()
    mean_milage = df['mileage'].mean() # since the data is almost normally distributed, mean is a better measure of central 

    return median_year, median_engine, median_km_driven, mean_milage


def get_X(user_year, user_engine, user_km_driven, user_mileage):
    default_year, default_engine, default_km_driven, default_mileage = getDefaultValue()

    # fill with default values if user input is empty    
    user_year = user_year if user_year else default_year
    user_engine = user_engine if user_engine else default_engine
    user_km_driven = user_km_driven if user_km_driven else default_km_driven
    user_mileage = user_mileage if user_mileage else default_mileage

    return user_year, user_engine, user_km_driven, user_mileage

@app.callback(
    Output('output-div', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('input-1', 'value'),
     State('input-2', 'value'),
     State('input-3', 'value'),
     State('input-4', 'value')]
)
def update_output(n_clicks, user_year, user_engine, user_km_driven, user_mileage):
    user_year, user_engine, user_km_driven, user_mileage = get_X(user_year, user_engine, user_km_driven, user_mileage)
    
    # [   29999.,   260000.,   450000.,   680000., 10000000.]
    prediction_label = {
        0: 'Cheap',
        1: 'Moderate',
        2: 'Expensive',
        3: 'Very Expensive'
    }

    prediction_range = {
        0: '29999 to 260000',
        1: '260000 to 450000',
        2: '450000 to 680000',
        3: '680000 to 10000000'
    }
    
    try:
        user_year = float(user_year)
        user_engine = float(user_engine)
        user_km_driven = float(user_km_driven)
        user_mileage = float(user_mileage)
    except ValueError:
        return "Please enter a valid input."

    if n_clicks > 0:
        pred_val = prediction(user_year, user_engine, user_km_driven, user_mileage)
        return f"**Predicted selling category**: {prediction_label[pred_val[0]]}  \n**Price Range lies between**: {prediction_range[pred_val[0]]}"
    return ""

if __name__ == '__main__':
    # before running the app, downloading the model from mlflow
    # from utils import load_mlflow
    # load_mlflow(stage="Production")
    app.run(host='0.0.0.0', port = 80, debug=True)
