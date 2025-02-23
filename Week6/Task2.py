from NeuralNetwork import NeuralNetwork
import numpy as np
import pandas as pd

def preprocess(dataset):
    data = pd.read_csv(dataset)
    # Drop missing age
    data = data.dropna(subset=['Age'])
    # Encode 'Sex' column: Male -> 1, Female -> 0
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
    # Select relevant features
    features = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values.T
    labels = data[['Survived']].values.T

    return features, labels

def normalize(X):
    min_array = np.min(X, axis=1, keepdims=True)
    max_array = np.max(X, axis=1, keepdims=True)
    return (X - min_array) / (max_array - min_array)

def main():
    X, Y = preprocess('titanic.csv')
    X = normalize(X)
    nn = NeuralNetwork(input_size=6, hidden_layers=[5, 4, 3, 2], output_size=1, learning_rate=0.01, epochs=40000, task_type="classification")

    before_training = nn.predict(X)
    before_training_acc = np.sum(before_training == Y) / Y.shape[1]
    print("------ Task 2 ------")
    print("------ Before Training ------")
    print(f"{before_training_acc*100:} %")

    print("------ Start Training ------")
    nn.train(X, Y)
    after_training = nn.predict(X)
    after_training_acc = np.sum(after_training == Y) / Y.shape[1]
    print("------ Start Evaluating ------")
    print(f"{after_training_acc*100:} %")

if __name__ == "__main__":
    main()