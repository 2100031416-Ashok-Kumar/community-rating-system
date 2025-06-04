import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import RobertaModel, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
import os

from dataset import CommentDataset, load_data
from model import CommentRatingModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RoBERTaWithMetadata(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout)
        self.metadata_fc = nn.Linear(3, 64)
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # Regression output
        )

    def forward(self, input_ids, attention_mask, metadata):
        roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        meta_out = torch.relu(self.metadata_fc(metadata))
        combined = torch.cat([roberta_out, meta_out], dim=1)
        return self.classifier(combined)

def train(model, train_loader, val_loader, epochs=5, lr=2e-5, checkpoint_path="models/best_model.pt"):
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            outputs = model(input_ids, attention_mask, metadata)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                metadata = batch["metadata"].to(device)
                labels = batch["label"].to(device).unsqueeze(1)

                outputs = model(input_ids, attention_mask, metadata)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print("✅ Saved best model")

if __name__ == "__main__":
    df = pd.read_csv("data/final_dataset.csv")
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    train_dataset = CommentDataset(train_df)
    val_dataset = CommentDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = RoBERTaWithMetadata().to(device)
    os.makedirs("models", exist_ok=True)
    train(model, train_loader, val_loader)
