import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

class CommentDataset(Dataset):
    def __init__(self, dataframe, max_len=128):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["helpfulness_score"].tolist()
        self.toxicity = dataframe["toxicity_score"].tolist()
        self.readability = dataframe["readability_score"].tolist()
        self.anomaly = dataframe["is_anomalous"].tolist()

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Metadata features: concatenate scalar values
        metadata = torch.tensor([
            self.toxicity[idx],
            self.readability[idx],
            self.anomaly[idx]
        ], dtype=torch.float)

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "metadata": metadata,
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        }
