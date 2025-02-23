from NeuralNetwork import NeuralNetwork
import numpy as np
import pandas as pd

def preprocess(dataset):
    data = pd.read_csv(dataset)
    weight_mean = np.mean(data["Weight"])
    weight_std = np.std(data["Weight"])
    # Standardize Height & Weight
    data["Height"] = (data["Height"] - data["Height"].mean()) / data["Height"].std()
    data["Weight"] = (data["Weight"] - data["Weight"].mean()) / weight_std
    # One-hot encode Gender
    data["Male"] = (data["Gender"] == "Male").astype(int) # Convert True/False to 1/0
    data["Female"] = (data["Gender"] == "Female").astype(int)
    # Select features
    features = data[["Male", "Female", "Height"]].values.T
    weights = data[["Weight"]].values.T

    return features, weights, weight_mean, weight_std

def main():
    X, Y, weight_mean, weight_std = preprocess('gender-height-weight.csv')
    nn = NeuralNetwork(input_size=3, hidden_layers=[10, 5], output_size=1, learning_rate=0.01, epochs=15000, task_type="regression")
    before_training = nn.predict(X)

    # De-standardize predicted Weight values
    before_training_unscaled = before_training * weight_std + weight_mean
    Y_unscaled = Y * weight_std + weight_mean

    # RMSE
    before_training_error = np.sqrt(np.mean((before_training_unscaled - Y_unscaled) ** 2))

    print("------ Task 1 ------")
    print("------ Before Training ------")
    print(f"Average Loss in Weight: {before_training_error} lbs")
    print("------ Start Training ------")
    nn.train(X, Y)
    after_training = nn.predict(X)

    # De-standardize predicted Weight values
    after_training_unscaled = after_training * weight_std + weight_mean
    # RMSE
    after_training_error = np.sqrt(np.mean((after_training_unscaled - Y_unscaled) ** 2))

    print("------ Start Evaluating ------")
    print(f"Average Loss in Weight: {after_training_error} lbs")


if __name__ == "__main__":
    main()