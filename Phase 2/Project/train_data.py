import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from gensim.models.doc2vec import Doc2Vec

MODEL = "doc2vec.model"
INFERRED_VEC = "inferred_vectors.npy"
SAMPLE_MODEL = "sample_input/sample.model"
SAMPLE_INFERRED_VEC = "sample_input/sample_inferred_vectors.npy"

class NeuralNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_model(model_path=MODEL, vectors_path=INFERRED_VEC):
    if os.path.exists(model_path) and os.path.exists(vectors_path):
        print(f"Loading model from: {model_path}")
        model = Doc2Vec.load(model_path)
        inferred_vectors = np.load(vectors_path, allow_pickle=True)
        return model, inferred_vectors
    else:
        raise FileNotFoundError("Model or vectors file not found.")

def split_data(inferred_vectors, labels, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(inferred_vectors, labels, test_size=test_size, random_state=42, stratify=labels)
    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_test = torch.tensor(y_test.values, dtype=torch.long)
    return X_train, X_test, y_train, y_test

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        # Train Phase
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        # Evaluating Phase
        model.eval()
        total_loss = 0
        first_match_count = 0
        second_match_count = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                total_samples += labels.size(0)

                top2 = torch.topk(outputs, k=2, dim=1).indices
                for i in range(labels.size(0)):
                    true_label = labels[i].item()
                    top_preds = top2[i].tolist()
                    if true_label == top_preds[0]:
                        first_match_count += 1
                        second_match_count += 1
                    elif true_label == top_preds[1]:
                        second_match_count += 1

        avg_test_loss = total_loss / len(test_loader)
        first_match = first_match_count / total_samples
        second_match = second_match_count / total_samples

        # === Print All Metrics ===
        print(f"======= Epoch {epoch+1} =======")
        print(f"Average Loss in Training Data: {avg_train_loss}")
        print(f"Average Loss in Test Data: {avg_test_loss}")
        print(f"First Match: {first_match}")
        print(f"Second Match: {second_match}")


def main():
    model, inferred_vectors = load_model()
    df = pd.read_csv("segmented_data2.csv", names=["Label", "Segmented"])
    labels = df["Label"].astype("category").cat.codes
    num_classes = len(set(labels))

    X_train, X_test, y_train, y_test = split_data(inferred_vectors, labels)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

    # Initialize the classification model
    input_dim = inferred_vectors.shape[1]
    classification_model = NeuralNet(input_dim, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classification_model.parameters(), lr=0.001)

    print("Training & Evaluating model...")
    train_and_evaluate(classification_model, train_loader, test_loader, criterion, optimizer, 30)

    torch.save(classification_model.state_dict(), "classification_model.pth")


if __name__ == '__main__':
    main()
