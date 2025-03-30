from app import prediction, get_X

feature_values = ['2015.0', '1248.0', '60000.0', '19.391961863322244']

# testing if model takes the expected input
def test_get_X():
    output = get_X(*feature_values)
    assert output == ('2015.0', '1248.0', '60000.0', '19.391961863322244'), f"Expected output {feature_values} but got {output}"

# testing if the output of the model has the expected shape
def test_prediction():
    output = prediction(2015.0, 1248.0, 60000.0, 19.391961863322244)
    assert output.shape == (1,), f"Expected output shape (1,) but got {output.shape}"