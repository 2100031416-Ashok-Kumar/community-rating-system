# dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

class CommentDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        comment = self.data.iloc[idx]['comment']
        label = self.data.iloc[idx]['label']
        encoding = self.tokenizer(
            comment,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data(path):
    df = pd.read_csv(path)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    dataset = CommentDataset(df, tokenizer)
    return dataset
