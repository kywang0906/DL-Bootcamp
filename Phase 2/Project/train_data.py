import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from gensim.models.doc2vec import Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

seed = 88
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Select device: GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device：{device}")

MODEL_PATH     = "doc2vec.model"
VECTORS_PATH   = "inferred_vectors.npy"
CLEANED_CSV    = "segmented_data_cleaned.csv"

class NeuralNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        return self.out(x)

# Load Doc2Vec model & inferred vectors
def load_doc2vec_and_vectors(model_path, vectors_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(vectors_path):
        raise FileNotFoundError(f"Inferred vectors not found: {vectors_path}")

    print(f"Doc2Vec loading: {model_path}")
    doc2vec = Doc2Vec.load(model_path)
    print(f"Inferred vectors loading: {vectors_path}")
    vectors = np.load(vectors_path, allow_pickle=True)
    return doc2vec, vectors

# Load segmented data and calculate weights
def load_data_and_weights(cleaned_csv, vectors):
    df = pd.read_csv(cleaned_csv, names=["Label", "Segmented"])
    labels = df["Label"].astype("category").cat.codes
    # Calculate class weights
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print("Creating TF-IDF features")
    tfidf = TfidfVectorizer(
        max_features=200,
        token_pattern=r"(?u)\b\w+\b"
    )
    X_tfidf = tfidf.fit_transform(df["Segmented"].tolist()) \
                   .toarray().astype(np.float32)
    print(f"TF-IDF shape: {X_tfidf.shape}")

    print("Merging Doc2Vec vectors & TF-IDF")
    X_doc2vec = vectors.astype(np.float32)
    X_combined = np.hstack([X_doc2vec, X_tfidf])
    print(f"Shape after merging: {X_combined.shape}")

    return X_combined, labels, class_weights

# Creating DataLoader
def create_dataloaders(vectors, labels, test_size=0.2, batch_size=128):
    X_train, X_test, y_train, y_test = train_test_split(
        vectors, labels, test_size=test_size,
        stratify=labels, random_state=seed
    )
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test  = torch.tensor(X_test,  dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.long).to(device)
    y_test  = torch.tensor(y_test.values,  dtype=torch.long).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader

# Train & evaluate model
def train_and_evaluate(model, train_loader, test_loader,
                       criterion, optimizer, scheduler,
                       epochs=100, patience=20):
    best_acc = 0
    no_improve = 0

    for epoch in range(1, epochs+1):
        # --- Training ---
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Testing ---
        model.eval()
        total_val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        avg_val_loss = total_val_loss / len(test_loader)
        acc = accuracy_score(all_labels, all_preds)

        print(f"======= Epoch {epoch} =======")
        print(f"Training Loss: {avg_train_loss} | "
              f"Test Loss: {avg_val_loss} | "
              f"Test Acc: {acc}")

        # Modify learning rate
        scheduler.step(acc)

        # Set early Stopping & save best model
        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            torch.save(model.state_dict(), "best_classification_model.pth")
            print(f"→ Best model saved (Acc: {best_acc})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping：連續 {patience} 輪無提升，停止訓練")
                break

    print(f"Training complete. Best test accuracy: {best_acc}")

if __name__ == '__main__':
    # Load Doc2Vec & inffered vectors
    doc2vec_model, inferred_vectors = load_doc2vec_and_vectors(
        MODEL_PATH, VECTORS_PATH
    )

    # Load data & weights
    X_combined, labels, class_weights = load_data_and_weights(
        CLEANED_CSV,
        inferred_vectors
    )

    # Create DataLoader
    train_loader, test_loader = create_dataloaders(
        X_combined, labels,
        batch_size=128, test_size=0.2
    )

    # Initialize model
    input_dim = X_combined.shape[1]
    num_classes = len(set(labels))
    model = NeuralNet(input_dim, num_classes).to(device)

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=5, verbose=True
    )

    # Start training and testing
    train_and_evaluate(
        model, train_loader, test_loader,
        criterion, optimizer, scheduler,
        epochs=100, patience=20
    )