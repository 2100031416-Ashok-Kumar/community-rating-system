import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score
import torch
from train_model import HelpfulnessModel, tokenizer
import numpy as np
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "../models/helpfulness_model.pt"

def evaluate():
    df = pd.read_csv("../data/sample_comments.csv")

    df['helpfulness'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min() + 1e-8)

    meta_cols = ['token_len', 'toxicity', 'readability', 'political_mentions',
                 'health_mentions', 'science_mentions', 'engagement', 'time_since_posted']
    X_meta = df[meta_cols].fillna(0).values

    texts = df['clean_text'].tolist()
    targets = df['helpfulness'].values

    dataset = CommentDataset(texts, X_meta, targets)
    dataloader = DataLoader(dataset, batch_size=16)

    model = HelpfulnessModel("roberta-base", X_meta.shape[1])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    preds = []
    actuals = []

    with torch.no_grad():
        for input_ids, attention_mask, meta_features, targets in dataloader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            meta_features = meta_features.to(DEVICE)

            outputs = model(input_ids, attention_mask, meta_features)
            preds.extend(outputs.cpu().numpy())
            actuals.extend(targets.numpy())

    auc = roc_auc_score(actuals, preds)
    k = 100
    df_eval = pd.DataFrame({'pred': preds, 'actual': actuals})
    df_eval_sorted = df_eval.sort_values('pred', ascending=False).head(k)
    precision_at_k = (df_eval_sorted['actual'] > 0.5).mean()

    print(f"AUC: {auc:.4f}")
    print(f"Precision@{k}: {precision_at_k:.4f}")

if __name__ == "__main__":
    evaluate()
