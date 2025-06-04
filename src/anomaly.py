import joblib
import numpy as np

def load_anomaly_model(path='models/anomaly_detector.pkl'):
    return joblib.load(path)

def is_anomalous(features, model):
    score = model.decision_function([features])[0]
    return model.predict([features])[0] == -1, score
