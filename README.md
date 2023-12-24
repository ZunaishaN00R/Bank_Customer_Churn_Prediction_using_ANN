# Churn Prediction Neural Network
Bank Customer Churn Prediction using Artificial Neural Network

## Overview

This code implements a neural network for predicting customer churn using a dataset from a banking scenario. The dataset includes various features such as credit score, geography, gender, age, tenure, balance, and more. The neural network is built using TensorFlow's Keras API.

## Data Preprocessing

- Irrelevant columns (`RowNumber`, `CustomerId`, `Surname`) are dropped from the dataset.
- Categorical variables are encoded, with `Gender` being label-encoded and `Geography` one-hot encoded.
- Data distribution and relationships are visualized using Seaborn.

## Data Splitting

The dataset is split into training and testing sets for model evaluation.

## Model Architecture

The neural network consists of three layers:
- Input layer with 64 neurons and ReLU activation.
- Hidden layer with 32 neurons and ReLU activation.
- Output layer with 1 neuron and Sigmoid activation for binary classification.

The model is compiled using binary crossentropy loss and the Adam optimizer.

## Model Training

The model is trained on the training set for 10 epochs with a batch size of 32. The training process is visualized using training and validation loss and accuracy.

## Model Evaluation

The trained model is evaluated on the test set. Precision, recall, and F1-score are computed using scikit-learn's `classification_report`.

## Results

The model achieves an accuracy of 79% on the test set. However, caution is advised as precision, recall, and F1-score for the positive class (Exited=1) are low, indicating potential issues with the model's ability to predict customer churn.

## Usage

1. Install required libraries: `pandas`, `seaborn`, `matplotlib`, `scikit-learn`, `tensorflow`.
2. Load the dataset using `pandas.read_csv`.
3. Preprocess the data by dropping irrelevant columns and encoding categorical variables.
4. Visualize data distribution and relationships.
5. Split the data into training and testing sets.
6. Build, compile, and train the neural network using TensorFlow and Keras.
7. Evaluate the model on the test set and analyze the results.

Feel free to customize the code for your specific use case and dataset.
