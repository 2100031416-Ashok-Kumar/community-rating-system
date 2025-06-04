import torch
import torch.nn as nn
from transformers import RobertaModel

class RobertaWithMetadata(nn.Module):
    def __init__(self, model_name='roberta-base', metadata_dim=3, dropout=0.3):
        super(RobertaWithMetadata, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        hidden_size = self.roberta.config.hidden_size  # usually 768
        self.metadata_dim = metadata_dim

        # Final classifier takes RoBERTa CLS output + metadata
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + metadata_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)  # single score output (e.g., helpfulness)
        )

    def forward(self, input_ids, attention_mask, metadata):
        roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = roberta_out.pooler_output  # [CLS] token representation

        combined = torch.cat((cls_output, metadata), dim=1)
        out = self.classifier(self.dropout(combined))
        return out.squeeze(1)  # final regression output
