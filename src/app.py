import gradio as gr
import torch
from transformers import RobertaTokenizer
from src.model import CommentClassifier
from src.utils import preprocess_comment, extract_metadata_features
from src.anomaly import load_anomaly_model


# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = CommentClassifier()
model.load_state_dict(torch.load("models/best_model.pt", map_location=torch.device("cpu")))
model.eval()

# Load Isolation Forest model
anomaly_model = load_anomaly_model()

def predict_comment(text):
    if not text.strip():
        return "Enter a valid comment!", 0, 0, 0, 0

    # Preprocess
    input_text = preprocess_comment(text)
    meta_features = extract_metadata_features(text)
    meta_tensor = torch.tensor(meta_features, dtype=torch.float).unsqueeze(0)

    # Tokenize
    inputs = tokenizer(
        input_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            meta_features=meta_tensor
        )
        probs = torch.softmax(outputs, dim=1).squeeze().numpy()

    labels = ["Negative", "Neutral", "Positive"]
    prediction = labels[probs.argmax()]
    sentiment_score = round(probs[2], 2)
    
    # Fake/toxic/helpfulness scores (simple heuristics)
    toxicity = float(meta_features[0])  # e.g., from Detoxify or rule-based
    helpfulness = float(meta_features[2])
    anomaly_score = anomaly_model.decision_function([meta_features])[0]

    return prediction, sentiment_score, round(toxicity, 2), round(helpfulness, 2), round(anomaly_score, 2)

# Gradio UI
iface = gr.Interface(
    fn=predict_comment,
    inputs=gr.Textbox(lines=4, label="Enter a Comment"),
    outputs=[
        gr.Textbox(label="Predicted Sentiment"),
        gr.Number(label="Sentiment Score (0-1)"),
        gr.Number(label="Toxicity Score"),
        gr.Number(label="Helpfulness Score"),
        gr.Number(label="Anomaly Score")
    ],
    title="🧠 Comment Quality Analyzer",
    description="Paste any social media comment and get its sentiment, toxicity, helpfulness, and anomaly scores using a RoBERTa-based ML model.",
    allow_flagging="never",
)

if __name__ == "__main__":
    iface.launch()
