import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def clean_data(data):
    x_df = data.to_pandas_dataframe().dropna()
    # diagnosis converted to binary categories
    y_df = x_df.pop("diagnosis").apply(lambda s: 1 if s == "M" else 0)
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--layer_size', type=int, default=32, help="Hidden layer sizes")
    parser.add_argument('--activation', type=str, default='relu', help="Activation function for the hidden layers")
    parser.add_argument('--C', type=float, default=0.0001, help="L2 regularization strength")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum iteration number")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Hidden layer size", np.int(args.layer_size))
    run.log("Activation function", args.activation)
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    ds =  ds = TabularDatasetFactory.from_delimited_files(
        path="./data/CancerData.csv", 
    )
    x, y = clean_data(ds)

    # TODO: Split data into train and test sets.

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = MLPClassifier(
        hidden_layer_sizes=args.layer_sizes, 
        activation=args.activation, solver='sgd', 
        alpha=args.C, 
        max_iter=args.max_iter
    ).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()