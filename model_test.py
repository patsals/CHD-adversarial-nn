from Adversarial_Model_Module import AdversarialModel
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np

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

    parser = argparse.ArgumentParser(description="Run the Python script with a flag")

    parser.add_argument('--lambda_tradeoff', type=float, help='Activate the flag', default=0.1)
    parser.add_argument('--gbt_retrain', type=float, help='Activate the flag', default=5)
    parser.add_argument('--epochs', type=float, help='Activate the flag', default=100)
    parser.add_argument('--learning_rate', type=float, help='Activate the flag', default=0.001)
    parser.add_argument('--sensitive_attribute', type=str, help='Activate the flag', default='Gender')
    
    args = parser.parse_args()

    lamdba_tradeoff = args.lambda_tradeoff
    gbt_retrain = args.gbt_retrain
    epochs = args.epochs
    learning_rate = args.learning_rate
    sensitive_attribute = args.sensitive_attribute

    print(args)


    df = pd.read_parquet('data/nhanes_data_processed.parquet')
    df.head()

    X, y = process_data(data_df=df, target='Coronary heart disease')

    # using pd.get_dummies() alters the original sensitive column's name
    new_sensitive_column_name = [col for col in X.columns if sensitive_attribute in col][0]

    label_column = 'Coronary heart disease'
    feature_columns = [col for col in df.columns if col != label_column]
    

    # # Convert Pandas DataFrame to a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        X.values.astype(np.float32),  # Features
        X[new_sensitive_column_name].values.reshape(-1, 1).astype(np.int32),  # Sensitive Feature
        y.values.astype(np.int32)  # Ensure labels are integers
    ))
    # # Define batch size
    batch_size = 16  

    # # Shuffle before batching
    buffer_size = len(df)  # Ideally, use the dataset size as the buffer
    shuffled_dataset = dataset.shuffle(buffer_size, seed=42).batch(batch_size)

    # BUT WE ALSO NEED TO SCALE THE DATA AND SPLIT IT INTO TRAIN TEST..
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # X_val =  scaler.transform(X_val)


    model = AdversarialModel(input_dim=41, # we can infer input_dim by len(axis=1) of dataset
                             sensitive_attr=sensitive_attribute,
                             lambda_tradeoff=lamdba_tradeoff, 
                             GBT_retrain=gbt_retrain, 
                             epochs=epochs,
                             learning_rate=learning_rate)
    

    model.fit(shuffled_dataset)