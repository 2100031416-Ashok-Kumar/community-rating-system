import numpy as np
import re
import string
from detoxify import Detoxify

detox_model = Detoxify('original')

def preprocess_comment(text):
    text = re.sub(r"http\S+", "", text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def extract_metadata_features(text):
    toxicity = detox_model.predict(text)["toxicity"]
    word_count = len(text.split())
    readability = min(1.0, word_count / 50)  # Normalize to [0,1]
    engagement = min(1.0, sum(1 for w in text.split() if len(w) > 6) / word_count) if word_count else 0
    return np.array([toxicity, readability, engagement], dtype=np.float32)
