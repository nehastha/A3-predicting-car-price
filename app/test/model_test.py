# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# from app import get_X, prediction
# # from app.logisticRegression import Normal

# # from app.logisticRegression import LogisticRegression

# feature_values = ['2015.0', '1248.0', '60000.0', '19.391961863322244']


# # testing if model takes the expected input
# def test_get_X():
#     output = get_X(*feature_values)
#     assert output == ('2015.0', '1248.0', '60000.0', '19.391961863322244'), f"Expected output {feature_values} but got {output}"

# # testing if the output of the model has the expected shape
# def test_prediction():
#     output = prediction(2015.0, 1248.0, 60000.0, 19.391961863322244)
#     assert output.shape == (1,), f"Expected output shape (1,) but got {output.shape}"

# import os
# import pickle
# import cloudpickle
# import pandas as pd
# import pytest

# # Define the paths for your model and scaler
# model_path = "model/a3_model.model"
# scalar_model_path = "model/a3_scaler.model"
# # with open("model/a3_model.model", "rb") as scaler_file:
# #     scaler_model = pickle.load(scaler_file)

# # with open("model/a3_model.model", "rb") as f:
# #     model_path = cloudpickle.load(f)

# # Sample test data
# test = pd.DataFrame([['2015.0', '1248.0', '60000.0', '19.391961863322244']],
#                     columns=[ 'year', 'km_driven', 'mileage', 'engine'])

# # Load the scaler using pickle
# with open(scalar_model_path, 'rb') as f:
#     scaler_model = pickle.load(f)

# # Test function to check if the model file exists
# def test_input():
#     assert os.path.exists(model_path), f"Model file not found at: {model_path}"

#     # Load the model using pickle
#     # with open(model_path, 'rb') as f:
#     #     model = pickle.load(f)
#     with open(model_path, 'rb') as f:
#         model = cloudpickle.load(f)

#     try:
#         # Apply scaling transformation to the test data
#         data = scaler_model.transform(test)
#         predicted_value = model.predict(data)
#     except:
#         # Raise an assertion error if shape is incorrect
#         assert False, "Shape is incorrect or error occurred during prediction."

# # Test function to check the shape and output of the prediction
# def test_predicted_shape():
#     assert os.path.exists(model_path), f"Model file not found at: {model_path}"

#     # Load the model using pickle
#     # with open(model_path, 'rb') as f:
#     #     model = pickle.load(f)
#     with open(model_path, 'rb') as f:
#         model = cloudpickle.load(f)

#     # Apply scaling transformation to the test data
#     data = scaler_model.transform(test)

#     # Get the predicted value
#     predicted_value = model.predict(data)

#     # Check if the predicted value is within the expected range (0 to 3)
#     assert predicted_value[0] in [0, 1, 2, 3], f"Unexpected predicted value: {predicted_value[0]}"

#     # Assert that the predicted value has the correct shape
#     assert predicted_value.shape == (1,), f"Expected output shape (1,), but got {predicted_value.shape}"

import pytest
import pickle
import pandas as pd
import numpy as np
import os
import cloudpickle


# scaler = joblib.load("app/model/scaler.dump")
scaler = pickle.load(open("model/a3_scaler.model", 'rb'))
with open("model/a3_model.model", "rb") as f:
    model = cloudpickle.load(f)


sample_input = pd.DataFrame({
    'engine': [1500],
    'km_driven': [85],
    'mileage': [20],
    'year': [2017]
})


def test_model_accepts_input():
    """Test if the model accepts input and does not throw an error"""
    try:
        scaled = scaler.transform(sample_input)
        model.predict(scaled)
        passed = True
    except Exception as e:
        passed = False
    assert passed, "Model failed to accept input format"


def test_model_output_shape():
    """Test if the model output shape is (1,)"""
    scaled = scaler.transform(sample_input)
    prediction = model.predict(scaled)
    assert prediction.shape == (1,), f"Expected shape (1,), but got {prediction.shape}"


