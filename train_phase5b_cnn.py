"""
train_phase5b_cnn.py
--------------------
Phase 5B: 1D CNN classifier for dynamometer cards.

Input:
  - data/processed/processed_cards_shape.npy  [400, 200, 2]
  - data/processed/processed_metadata.csv

Target:
  - condition_label (Normal, Rod Floating, Fluid Pound, Gas Interference)

Split:
  - Well-level train/val split from processed_metadata.csv (split column)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase5"

CLASS_ORDER = ["Normal", "Rod Floating", "Fluid Pound", "Gas Interference"]
LABEL2IDX = {label: i for i, label in enumerate(CLASS_ORDER)}
IDX2LABEL = {i: label for label, i in LABEL2IDX.items()}


class DynaCardDataset(Dataset):
    def __init__(self, cards, labels):
        # cards shape: [N, 200, 2] -> transpose to [N, 2, 200] for Conv1d
        self.cards = torch.tensor(cards.transpose(0, 2, 1), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cards[idx], self.labels[idx]


class DynaCardCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # (B, 32, 100)

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # (B, 64, 50)

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # (B, 128, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    cards_shape = np.load(DATA_DIR / "processed_cards_shape.npy")
    metadata = pd.read_csv(DATA_DIR / "processed_metadata.csv")

    assert len(cards_shape) == len(metadata) == 400

    train_mask = (metadata["split"] == "train").values
    val_mask = (metadata["split"] == "val").values

    labels = np.array([LABEL2IDX[c] for c in metadata["condition_label"]])

    X_train, y_train = cards_shape[train_mask], labels[train_mask]
    X_val, y_val = cards_shape[val_mask], labels[val_mask]

    train_dataset = DynaCardDataset(X_train, y_train)
    val_dataset = DynaCardDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 2. Compute class weights for balanced loss
    class_counts = np.bincount(y_train, minlength=4)
    class_weights = 1.0 / (class_counts / class_counts.sum())
    class_weights = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float32)

    # 3. Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynaCardCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 4. Training loop
    epochs = 60
    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Validation evaluation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)

        val_f1 = f1_score(y_val, val_preds, average="macro")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()

    # Load best weights
    model.load_state_dict(best_state)
    model.eval()

    # Final validation predictions
    val_preds = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)

    val_preds = np.array(val_preds)
    y_val_str = [IDX2LABEL[i] for i in y_val]
    val_preds_str = [IDX2LABEL[i] for i in val_preds]

    # Metrics
    val_acc = accuracy_score(y_val_str, val_preds_str)
    val_bal_acc = balanced_accuracy_score(y_val_str, val_preds_str)
    val_macro_f1 = f1_score(y_val_str, val_preds_str, average="macro")
    report = classification_report(y_val_str, val_preds_str, labels=CLASS_ORDER, digits=4, zero_division=0)
    cm = confusion_matrix(y_val_str, val_preds_str, labels=CLASS_ORDER)

    print("=" * 60)
    print("TEJAS Phase 5B — 1D CNN Classification Results")
    print("=" * 60)
    print(f"Validation Accuracy:          {val_acc:.4f}")
    print(f"Validation Balanced Accuracy: {val_bal_acc:.4f}")
    print(f"Validation Macro F1:          {val_macro_f1:.4f}")
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    # Save model and artifacts
    torch.save(model.state_dict(), MODELS_DIR / "cnn_phase5b.pt")
    
    cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)
    cm_df.to_csv(RESULTS_DIR / "cnn_confusion_matrix.csv")

    with open(RESULTS_DIR / "cnn_classification_report.txt", "w") as f:
        f.write("TEJAS Phase 5B — 1D CNN Evaluation Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Validation Accuracy:          {val_acc:.4f}\n")
        f.write(f"Validation Balanced Accuracy: {val_bal_acc:.4f}\n")
        f.write(f"Validation Macro F1:          {val_macro_f1:.4f}\n\n")
        f.write(report + "\n")
        f.write("Confusion Matrix (rows=true, cols=pred):\n")
        f.write(cm_df.to_string() + "\n")

    print("\nSaved model to models/cnn_phase5b.pt")
    print("Saved results to results/phase5/cnn_classification_report.txt and cnn_confusion_matrix.csv")


if __name__ == "__main__":
    main()
