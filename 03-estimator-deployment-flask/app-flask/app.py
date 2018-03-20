import os
import tensorflow as tf
import numpy as np

# Check that we have correct TensorFlow version installed
tf_version = tf.__version__
print("TensorFlow version: {}".format(tf_version))
assert "1.5" <= tf_version, "TensorFlow r1.5 or later is needed"

from flask import Flask
app = Flask(__name__)


class Model:

    def __init__(self, model_directory):
        self.model_directory = model_directory
        self._predict_fn = self._load_model()

    def _load_model(self):
        return tf.contrib.predictor.from_saved_model(self.model_directory, signature_def_key='predict')

    def predict(self, features):
        return self._predict_fn(features)


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/predict",  methods=['GET', 'POST'])
def predict():
    # lazy load model
    model = Model('../saved_model/1521558615')

    # extract parameters from request
    prediction_input_single = { 
        'PctUnder18': [23.9],
        'PctOver65': [17.6],
        'PctFemale': [50.0],
        'PctWhite': [0.965],
        'PctBachelors': [12.7],
        'PctDem': [0.3227832512315271],
        'PctGop': [0.6545566502463054]
    }   

    # run prediction and return result
    prediction_result = model.predict(prediction_input_single)
    print(prediction_result)
    return str(prediction_result['predictions'])

