import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_decision_forests as tfdf
import tf_keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score,accuracy_score, precision_score, recall_score, f1_score

from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
    
def model_assessment(predictions, actuals):
    
    #Overall
    overall_accuracy = accuracy_score(actuals, predictions)
    overall_recall = recall_score(actuals, predictions)
    overall_precision = precision_score(actuals, predictions)

    results = pd.DataFrame({
        'Group': ['Overall'],
        'Accuracy': [overall_accuracy],
        'Recall': [overall_recall],
        'Precision': [overall_precision]
    })

    return results



def data_processor(data, batch_size):

    # Define features to standardize (Z-score)
    standardize_features = ["Red blood cells", "Hemoglobin", "Albumin", "Protein"]

    # Define features to normalize (Min-Max)
    normalize_features = ["ALT", "AST", "LDH", "Bilirubin", "Triglycerides", "Glucose"]

    # Categorical Columns
    categorical_cols = ['Diabetes', 'Blood related diabetes', 'Vigorous-work']


    # Data Processing Helper Functions
    scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    encoder = OneHotEncoder(sparse=False, drop="first")  


    # Binarize Binary Categorical Columns
    data['Gender'] =  data['Gender'].mask(data['Gender'] == 'Male', 1).mask(data['Gender'] == 'Female', 0)
    data['Coronary heart disease'] = data['Coronary heart disease'].mask(data['Coronary heart disease'] == 'Yes', 1).mask(data['Coronary heart disease'] == 'No', 0)

    # Separate Data to Features and Labels
    X = data.drop('Coronary heart disease', axis = 1)
    y = data['Coronary heart disease']

    
    # Split Data into Train, Validatin, and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # One-hot Encoding and Standardization/Normalization of Data
    X_train[standardize_features] = scaler.fit_transform(X_train[standardize_features])
    X_val[standardize_features] = scaler.transform(X_val[standardize_features])
    X_test[standardize_features] = scaler.transform(X_test[standardize_features])

    X_train[normalize_features] = minmax_scaler.fit_transform(X_train[normalize_features])
    X_val[normalize_features] = minmax_scaler.transform(X_val[normalize_features])
    X_test[normalize_features] = minmax_scaler.transform(X_test[normalize_features])

    # Convert to DataFrame with correct column names
    X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
    X_val_encoded = encoder.transform(X_val[categorical_cols])
    X_test_encoded = encoder.transform(X_test[categorical_cols])

    train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X_train.index)
    val_encoded_df = pd.DataFrame(X_val_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X_val.index)
    test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X_test.index)

    # Drop original categorical columns and concatenate the new one-hot encoded features
    X_train = X_train.drop(columns=categorical_cols).reset_index(drop=True)
    X_val = X_val.drop(columns=categorical_cols).reset_index(drop=True)
    X_test = X_test.drop(columns=categorical_cols).reset_index(drop=True)

    X_train = pd.concat([X_train, train_encoded_df], axis=1)
    X_val = pd.concat([X_val, val_encoded_df], axis=1)
    X_test = pd.concat([X_test, test_encoded_df], axis=1)

    # Convert Pandas DataFrame to a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((
    X_train.to_numpy().astype(np.float32),  # Features
    X_train['Gender'].to_numpy().reshape(-1, 1).astype(np.int32),  # Sensitive Feature
    y_train.to_numpy().values.astype(np.int32)  # Ensure labels are integers
    ))

    # Batch Training Set
    buffer_size = len(X_train)  # Ideally, use the dataset size as the buffer
    batched_dataset = dataset.shuffle(buffer_size, seed=42).batch(batch_size)

    return batched_dataset, X_val, y_val, X_test, y_test

