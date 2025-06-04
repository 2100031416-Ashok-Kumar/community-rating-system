# 🧠 Community Rating System for Social Posts

A machine learning-powered system that evaluates **community comments or notes** on public posts for **credibility, helpfulness, and factual consistency**, while actively resisting manipulation like spam voting or coordinated misinformation.

---

## 🔍 Project Overview

This project builds a full-stack pipeline that:

- Processes public comments (e.g., Reddit, StackOverflow)
- Extracts features like sentiment, toxicity, readability, and named entities
- Scores notes/comments for **credibility** and **helpfulness**
- Detects manipulation attempts (e.g., bot voting, duplicate spam)
- Provides a web interface to interactively test the system

---

## 🚀 Features

- ✅ Transformer-based helpfulness scoring (e.g., RoBERTa)
- ✅ Anomaly detection using Isolation Forest
- ✅ Sentiment and toxicity classification
- ✅ Interactive UI with Gradio or Streamlit
- ✅ REST API using FastAPI
- ✅ Dockerized and deployable on Hugging Face Spaces or Render

---

## 🗂️ Project Structure

```plaintext
.
├── data/
│   └── sample_comments.csv          # Input dataset
├── models/                          # Saved model checkpoints
├── src/
│   ├── app.py                       # Web interface
│   ├── anomaly.py                   # Anomaly detection logic
│   ├── data_preprocessing.py        # Cleaning, tokenization, features
│   ├── dataset.py                   # PyTorch dataset logic
│   ├── evaluate.py                  # Evaluation metrics
│   ├── main.py                      # Entry script
│   ├── manipulation_detect.py       # Bot/spam detection
│   ├── model.py                     # Helpfulness scoring model
│   ├── train_model.py               # Training loop
│   └── utils.py                     # Helper functions
├── requirements.txt                 # Dependencies
├── dockerfile                       # Container setup
└── README.md                        

## 📦 Setup Instructions
🔧 1. Install Dependencies
- git clone https://github.com/yourusername/community-rating-system.git
- cd community-rating-system
- python -m venv env
- source env/bin/activate  # or .\env\Scripts\activate on Windows
- pip install -r requirements.txt

## 🧪 2. Train the Helpfulness Model
- python src/train_model.py

## 🧬 3. Run Anomaly Detection
- python src/manipulation_detect.py

## 🌐 4. Launch Web Interface
- python src/app.py
or
- uvicorn src.main:app --reload

## 🔐 Manipulation Resistance
- This project uses:
- Isolation Forest to detect abnormal user behavior (e.g., sudden karma spikes)
- Text similarity analysis to flag near-duplicate spam
- (Optional) Adversarial training to improve robustness against vote brigading

## 🧠 ML Model Architecture
- Backbone: RoBERTa-base or BERT-base
- Input: comment_text, along with metadata like upvotes, length, time, etc.
- Output: A scalar score (0.0–1.0) representing helpfulness or credibility

Additional Features:
✅ Sentiment polarity

✅ Toxicity score (via Detoxify)

✅ Readability score (Flesch–Kincaid, Gunning Fog Index)

✅ Named Entity Types (political, scientific, health-related)


## 🧪 Evaluation
Key metrics used for evaluating the scoring model and resistance components:

🏆 ROC-AUC

🎯 Precision@K

🧩 Agreement with majority vote labels

📊 Confusion Matrix on manipulated vs. organic comment behavior

## 📦 Docker Deployment
- To build and run the container locally:
- docker build -t community-rating-system .
- docker run -p 7860:7860 community-rating-system


