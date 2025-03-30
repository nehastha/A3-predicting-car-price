import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.app import get_X, prediction
from app.logisticRegression import Normal

# from app.logisticRegression import LogisticRegression

feature_values = ['2015.0', '1248.0', '60000.0', '19.391961863322244']


# testing if model takes the expected input
def test_get_X():
    output = get_X(*feature_values)
    assert output == ('2015.0', '1248.0', '60000.0', '19.391961863322244'), f"Expected output {feature_values} but got {output}"

# testing if the output of the model has the expected shape
def test_prediction():
    output = prediction(2015.0, 1248.0, 60000.0, 19.391961863322244)
    assert output.shape == (1,), f"Expected output shape (1,) but got {output.shape}"

