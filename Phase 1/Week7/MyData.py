import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MyData(Dataset):
    def __init__(self, dataset, task_type):
        data = pd.read_csv(dataset)
        self.task_type = task_type # weight / titanic

        # Preprocess data
        if task_type == "titanic":
            # Drop missing Age
            data = data.dropna(subset=['Age'])

            # Sex -> 1 / 0
            data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})

            self.X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values.astype(np.float32)
            self.y = data[['Survived']].values.astype(np.float32)

        elif task_type == "weight":
            # Standardize height & weight
            self.mean_weight = np.mean(data["Weight"])
            self.std_weight = np.std(data["Weight"])
            data["Height"] = (data["Height"] - data["Height"].mean()) / data["Height"].std()
            data["Weight"] = (data["Weight"] - self.mean_weight) / self.std_weight

            # One-hot encode Gender
            data["Male"] = (data["Gender"] == "Male").astype(int)
            data["Female"] = (data["Gender"] == "Female").astype(int)

            self.X = data[["Male", "Female", "Height"]].values.astype(np.float32)
            self.y = data[["Weight"]].values.astype(np.float32)
        
        # Convert to tensor
        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y)
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]