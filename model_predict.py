#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 21:39:30 2023

@author: fathimazajel
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Function to read preprocessed data
def read_data(path):
    df = pd.read_csv(path)
    return df

# Function to select only numerical features
def select_numerical_features(df):
    df = df.select_dtypes(include=[np.number])
    return df

# Function to train the model and make predictions
def train_and_predict(train_df, test_df, target_column):
    # Splitting the features and target
    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]
    X_test = test_df  # Test data does not have the target column

    # Training the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Making predictions
    train_predictions = model.predict_proba(X_train)[:, 1]
    test_predictions = model.predict_proba(X_test)[:, 1]

    return train_predictions, test_predictions

if __name__ == "__main__":
    # Import the necessary module for preprocessing
    import preprocessing

    # Execute the function from the script
    preprocessing.process_data()

    # Preprocessing
    train_path = 'temp/training.csv'
    test_path = 'temp/test.csv'
    output_train_path = 'temp/preprocessed_train.csv'
    output_test_path = 'temp/preprocessed_test.csv'

    train_df, test_df = preprocessing.preprocess_data(train_path, test_path)

    preprocessing.write_data(train_df, output_train_path)
    preprocessing.write_data(test_df, output_test_path)

    # Read the data
    train_df = read_data('temp/preprocessed_train.csv')
    test_df = read_data('temp/preprocessed_test.csv')

    # Select only numerical features
    train_df = select_numerical_features(train_df)
    test_df = select_numerical_features(test_df)

    # Train the model and make predictions
    train_predictions, test_predictions = train_and_predict(train_df, test_df, 'isbadbuy')

    # Create DataFrames for the predictions
    train_predictions_df = pd.DataFrame({'RefId': train_df['refid'], 'IsBadBuy': train_predictions})
    test_predictions_df = pd.DataFrame({'RefId': test_df['refid'], 'IsBadBuy': test_predictions})

    # Save the predictions to CSV files
    train_predictions_df.to_csv('output/train_predictions.csv', index=False)
    test_predictions_df.to_csv('output/test_predictions.csv', index=False)
