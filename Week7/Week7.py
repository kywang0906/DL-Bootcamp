import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from NeuralNetwork import NeuralNetwork
from MyData import MyData

def train_model(model, dataset, loss_fn, optimizer, epochs):
    model.train()
    X, y = dataset.X, dataset.y
    for epoch in range(epochs):
        # Reset gradients before each epoch
        optimizer.zero_grad()
        y_pred = model(X).squeeze()
        loss = loss_fn(y_pred, y.squeeze())
        loss.backward()
        # Apply weight updates
        optimizer.step()

def evaluate_model(model, dataset, task_type):
    model.eval()
    X, y = dataset.X, dataset.y

    with torch.no_grad():
        y_pred = model(X)

    y_pred = y_pred.numpy()
    y = y.numpy()

    if task_type == "titanic":
        # Convert sigmoid output to binary classification
        y_pred = (y_pred > 0.5).astype(int)
        accuracy = np.mean(y_pred == y) * 100
        print(f"Accuracy: {accuracy}%")

    elif task_type == "weight":
        # Convert predictions back to original scale
        mean_weight = dataset.mean_weight
        std_weight = dataset.std_weight
        y_pred = y_pred * std_weight + mean_weight
        y = y * std_weight + mean_weight

        # Compute RMSE
        rmse = np.sqrt(np.mean((y_pred - y) ** 2))
        print(f"Average Loss in Weight: {rmse} lbs")

def main():
    learning_rate = 0.01
    epochs = 10000

    tasks = ["weight", "titanic"]
    for i, task_type in enumerate(tasks):   
        print(f'------ Task {i+1} ------')
        
        if task_type == "weight":
            dataset = MyData('gender-height-weight.csv', task_type)
        else: 
            dataset = MyData('titanic.csv', task_type)

        input_size = dataset.X.shape[1]
        output_size = 1
        hidden_layers = [10, 5]

        model = NeuralNetwork(input_size, hidden_layers, output_size, task_type)

        # Define loss function & optimizer
        loss_fn = nn.BCELoss() if task_type == "titanic" else nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        print("------ Before Training ------")
        evaluate_model(model, dataset, task_type)
        print("------ Start Training ------")
        train_model(model, dataset, loss_fn, optimizer, epochs)
        print("------ Start Evaluating ------")
        evaluate_model(model, dataset, task_type)


if __name__ == "__main__":
    main()
