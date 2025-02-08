# from Adversarial_Model_Module import AdversarialModel
from AMM_SGDClassifier import AdversarialModel
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import helpers
from sklearn.metrics import  f1_score
import matplotlib.pyplot as plt


categorical_cols = ['Diabetes', 'Blood related diabetes', 'Vigorous-work', 'Gender']

def process_data(data_df=pd.DataFrame, target=str):

    # One-hot encoding
    data_df = pd.get_dummies(data_df, columns=categorical_cols, drop_first=True)

    # Mapping target column to binary values
    data_df[target] = data_df[target] == data_df[target].unique()[0]

    #This creates split datasets for training, testing, and validation
    #Additionally it prepares the input data sets for model fitting and predicting
    X = data_df.drop(target, axis = 1)
    y = data_df[target]

    return X, y



if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
            )
        except RuntimeError as e:
            print(e)

    parser = argparse.ArgumentParser(description="Run the Python script with a flag")

    parser.add_argument('--lambda_tradeoff', type=float, help='Activate the flag', default=0.1)
    parser.add_argument('--gbt_retrain', type=float, help='Activate the flag', default=5)
    parser.add_argument('--epochs', type=float, help='Activate the flag', default=100)
    parser.add_argument('--learning_rate', type=float, help='Activate the flag', default=0.001)
    parser.add_argument('--sensitive_attribute', type=str, help='Activate the flag', default='Gender')
    
    args = parser.parse_args()

    lamdba_tradeoff = args.lambda_tradeoff
    gbt_retrain = args.gbt_retrain
    epochs = int(args.epochs)
    learning_rate = args.learning_rate
    sensitive_attribute = args.sensitive_attribute

    print(args)


    df = pd.read_parquet('data/nhanes_data_processed.parquet')

    batched_dataset, X_val, y_val, X_test, y_test = helpers.data_processor(df, 32)

    # X, y = process_data(data_df=df, target='Coronary heart disease')

    # # using pd.get_dummies() alters the original sensitive column's name
    # new_sensitive_column_name = [col for col in X.columns if sensitive_attribute in col][0]

    # label_column = 'Coronary heart disease'
    # feature_columns = [col for col in df.columns if col != label_column]
    

    # # # Convert Pandas DataFrame to a TensorFlow dataset
    # dataset = tf.data.Dataset.from_tensor_slices((
    #     X.values.astype(np.float32),  # Features
    #     X[new_sensitive_column_name].values.reshape(-1, 1).astype(np.int32),  # Sensitive Feature
    #     y.values.astype(np.int32)  # Ensure labels are integers
    # ))
    # # # Define batch size
    # batch_size = 16  

    # # # Shuffle before batching
    # buffer_size = len(df)  # Ideally, use the dataset size as the buffer
    # shuffled_dataset = dataset.shuffle(buffer_size, seed=42).batch(batch_size)

    # BUT WE ALSO NEED TO SCALE THE DATA AND SPLIT IT INTO TRAIN TEST..
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # X_val =  scaler.transform(X_val)


    model = AdversarialModel(input_dim=41, # we can infer input_dim by len(axis=1) of dataset
                             lambda_tradeoff=lamdba_tradeoff, 
                             epochs=epochs,
                             learning_rate=learning_rate)
    

    model.fit(batched_dataset)

    X_val_array = np.array(X_val).astype(np.float32) 
    X_val_tensor = tf.convert_to_tensor(X_val_array, dtype=tf.float32)
    raw_preds = model.predict(X_val_tensor, raw_probabilities=True)

    preds = (raw_preds >= 0.11).astype(int)
    preds = preds.flatten()
    y_val = y_val.astype(int)

    fairness_df = X_val.copy()
    fairness_df['Coronary heart disease'] = y_val

    print('Model assessment:')
    print(helpers.model_assessment(preds, y_val))

    print('Model fairness metrics:')
    print(helpers.fairness_metrics(fairness_df, preds))


    thresholds = np.arange(0.01,1, 0.05)
    f1_scores = []
    for t in thresholds:
        preds = (raw_preds >= t).astype(int)
        f1 = f1_score(y_val, preds)
        f1_scores.append(f1)

    optimal_threshold = thresholds[f1_scores.index(max(f1_scores))]
    print(f"Optimal threshold for F1-Score: {optimal_threshold}")

    plt.plot(thresholds, f1_scores)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('F1-Score vs. Threshold')
    plt.show()