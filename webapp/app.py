from flask import Flask, request, jsonify

import pandas as pd
from sklearn.externals import joblib
#import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def scale(payload):
    scaler=StandardScaler().fit(payload)
    return scaler.transform(payload)

@app.route("/")
def home():
    return "<h3>Skleran prediction container"

@app.route("/predict", methods=["POST"])
def predict():
    clf=joblib.load("boston_housing_prediction.joblib")
    inference_payload=pd.DataFrame(request.json)
    scaled_payload=scale(inference_payload)
    prediction=list(clf.predict(scaled_payload))
    return jsonify ({"Prediction":prediction})

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)