import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score,accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
    
def model_assessment(predictions, actuals):
    
    #Overall
    overall_accuracy = accuracy_score(actuals, predictions)
    overall_recall = recall_score(actuals, predictions)
    overall_precision = precision_score(actuals, predictions)
    overall_balanced_accuracy = balanced_accuracy_score(actuals, predictions)

    results = pd.DataFrame({
        'Group': ['Overall'],
        'Accuracy': [overall_accuracy],
        'Recall': [overall_recall],
        'Precision': [overall_precision],
        'Balanced Accuracy': [overall_balanced_accuracy]
    })

    return results

def fairness_metrics(input_df, predictions):

    binary_dataset = BinaryLabelDataset(df=input_df, 
                                    label_names=['Coronary heart disease'], 
                                    protected_attribute_names=['Gender'])

    # Create predictions dataset
    pred_df = input_df.drop('Coronary heart disease', axis = 1)
    pred_df['Coronary heart disease'] = predictions
    binary_predictions = BinaryLabelDataset(df=pred_df, 
                                            label_names=['Coronary heart disease'], 
                                            protected_attribute_names=['Gender'])

    # Compute metrics
    metric = ClassificationMetric(binary_dataset, binary_predictions, 
                                unprivileged_groups=[{'Gender': 1}], 
                                privileged_groups=[{'Gender': 0}]) 
    
    demographic_parity_difference = metric.statistical_parity_difference()
    equal_opportunity_difference = metric.equal_opportunity_difference()
    # predictive_parity = metric.statistical_parity_difference()
    disparate_impact = metric.disparate_impact()


    #Output Metrics in a Pandas DataFrame
    fairness_table = pd.DataFrame({
        'Metric': ['Demographic Parity Difference', 
                   'Equal Opportunity Difference',
                   #'Predictive Parity', 
                   'Disparate Impact'],
        'Value': [demographic_parity_difference, 
                  equal_opportunity_difference,
                #   predictive_parity, 
                  disparate_impact]
    })

    return fairness_table



def data_processor(data, batch_size, balance=False):
    # Define features to normalize (Min-Max)
    normalize_features = ['Weight', 'Body mass index', 'Systolic', 'Diastolic', 'Age',
       'Glycohemoglobin', 'Cholesterol',
       'High-density lipoprotein (HDL)', 'Albumin',
       'Alanine aminotransferase (ALT)', 'Aspartate aminotransferase (AST)',
       'Alkaline phosphatase (ALP)', 'Gamma-glutamyl transferase (GGT)',
       'Glucose', 'Iron', 'Lactate dehydrogenase (LDH)', 'Phosphorus',
       'Bilirubin', 'Protein', 'Triglycerides', 'Uric acid', 'Creatinine',
       'White blood cells', 'Basophils', 'Red blood cells', 'Hemoglobin',
       'Red blood cell width', 'Platelet count', 'Mean volume of platelets',
       'Moderate-work']

    # Categorical Columns
    categorical_cols = ['Diabetes', 'Blood related diabetes', 'Vigorous-work']


    # Data Processing Helper Functions
    minmax_scaler = MinMaxScaler()
    encoder = OneHotEncoder(drop="first")  


    # Binarize Binary Categorical Columns
    data['Gender'] =  data['Gender'].mask(data['Gender'] == 'Male', 1).mask(data['Gender'] == 'Female', 0)
    data['Coronary heart disease'] = data['Coronary heart disease'].mask(data['Coronary heart disease'] == 'Yes', 1).mask(data['Coronary heart disease'] == 'No', 0)

    
    # One-Hot Encoding Before Splitting
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    encoded_data = encoder.fit_transform(data[categorical_cols])

    # Convert to DataFrame
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols), index=data.index)

    # Drop original categorical columns and concatenate one-hot encoded columns
    data = data.drop(columns=categorical_cols).reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)
    data = pd.concat([data, encoded_df], axis=1)

    if balance:
        num_majority = len(data[data['Coronary heart disease'] == 0])
        num_minority = len(data[data['Coronary heart disease'] == 1])
        majority_minority_difference = num_majority - num_minority

        minority_df = data[data['Coronary heart disease'] == 1]
        additional_minority_indices = np.random.choice(len(minority_df), majority_minority_difference, replace=True)
        additional_minority_samples = minority_df.iloc[additional_minority_indices]

        data = pd.concat([data, additional_minority_samples], axis=0, ignore_index=True)

    # Separate Data to Features and Labels
    X = data.drop('Coronary heart disease', axis = 1)
    y = data['Coronary heart disease']

    
    # Split Data into Train, Validatin, and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # One-hot Encoding and Normalization of Data
    X_train[normalize_features] = minmax_scaler.fit_transform(X_train[normalize_features])
    X_val[normalize_features] = minmax_scaler.transform(X_val[normalize_features])
    X_test[normalize_features] = minmax_scaler.transform(X_test[normalize_features])



    # Convert Pandas DataFrame to a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((
    X_train.to_numpy().astype(np.float32),  # Features
    X_train['Gender'].to_numpy().reshape(-1, 1).astype(np.int32),  # Sensitive Feature
    y_train.to_numpy().astype(np.int32)  # Ensure labels are integers
    ))

    # Batch Training Set
    buffer_size = len(X_train)  # Ideally, use the dataset size as the buffer
    batched_dataset = dataset.shuffle(buffer_size, seed=42).batch(batch_size)

    return batched_dataset, X_train, y_train, X_val, y_val, X_test, y_test


def balance_dataset(data):
    num_majority = len(data[data['Coronary heart disease'] == 0])
    num_minority = len(data[data['Coronary heart disease'] == 1])
    majority_minority_difference = num_majority - num_minority

    minority_df = data[data['Coronary heart disease'] == 1]
    additional_minority_indices = np.random.choice(len(minority_df), majority_minority_difference, replace=True)
    additional_minority_samples = minority_df.iloc[additional_minority_indices]

    data = pd.concat([data, additional_minority_samples], axis=0, ignore_index=True)

    return data