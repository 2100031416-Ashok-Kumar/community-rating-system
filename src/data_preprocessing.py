import pandas as pd
import numpy as np
import re
from sklearn.ensemble import IsolationForest
from detoxify import Detoxify
import textstat

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load and preprocess
def load_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(subset=["text", "helpfulness_score"], inplace=True)
    return df

def preprocess_text(df):
    df['text'] = df['text'].apply(clean_text)

    # Toxicity score
    print("Computing toxicity scores...")
    toxicity_results = Detoxify('original').predict(df['text'].tolist())
    df['toxicity_score'] = toxicity_results['toxicity']

    # Readability score (lower is harder to read)
    print("Computing readability scores...")
    df['readability_score'] = df['text'].apply(lambda x: textstat.flesch_reading_ease(x))

    # Anomaly Detection
    print("Running anomaly detection...")
    meta_features = df[["toxicity_score", "readability_score"]].fillna(0)
    clf = IsolationForest(contamination=0.05, random_state=42)
    df['is_anomalous'] = clf.fit_predict(meta_features)
    df['is_anomalous'] = df['is_anomalous'].apply(lambda x: 1 if x == -1 else 0)

    return df

# For custom evaluation later
def compute_custom_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }
