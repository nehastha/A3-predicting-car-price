import pandas as pd
import cloudpickle
import os

base_path = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_path, "../../model/a3_model.model"), "rb") as f:
    model = cloudpickle.load(f)

sample_input = {
            'year': [2020],
            'engine': [1050],
            'km_driven': [7000],
            'mileage':[40]
        }

def test_model_accepts_input():
    """Test if the model accepts input and does not throw an error"""
    try:
        X = pd.DataFrame(sample_input, index=[0])
        X = X.to_numpy()
        model.predict(X)
        passed = True
    except Exception as e:
        passed = False
    assert passed, "Model failed to accept input format"


def test_model_output_shape():
    """Test if the model output shape is (1,)"""
    X = pd.DataFrame(sample_input, index=[0])
    X = X.to_numpy()
    prediction = model.predict(X)
    assert prediction.shape == (1,), f"Expected shape (1,), but got {prediction.shape}"
