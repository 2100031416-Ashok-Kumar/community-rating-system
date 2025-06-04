import torch.nn as nn
from transformers import RobertaModel

class CommentClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super(CommentClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size + 3, 1)  # +3 for metadata features

    def forward(self, input_ids, attention_mask, metadata_features):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        combined = torch.cat((pooled_output, metadata_features), dim=1)
        x = self.dropout(combined)
        return self.classifier(x)
