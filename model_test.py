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
import os

RESULTS_DIRECTORY = 'results'

def plot_metric(x, y, x_label, y_label, title, metric, plot_directory):
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    # Save the plot as a PNG file
    plot_file_path = os.path.join(plot_directory, metric+'.png')
    plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the Python script with a flag")

    parser.add_argument('--lambda_tradeoff', type=float, default=0.1)
    parser.add_argument('--epochs', type=float, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=str, default=32)
    parser.add_argument('--adv_model_type', type=str, default='logistic_regression')
    parser.add_argument('--patience', type=int, default=10)

    args = parser.parse_args()
    lamdba_tradeoff = args.lambda_tradeoff
    epochs = int(args.epochs)
    learning_rate = args.learning_rate
    batch_size = int(args.batch_size)
    adv_model_type = args.adv_model_type
    patience = args.patience
    
    print(args)

    df = pd.read_parquet('data/nhanes_data_processed.parquet')

    batched_dataset, X_val, y_val, X_test, y_test = helpers.data_processor(df, batch_size, balance=True)

    model = AdversarialModel(input_dim=41, # we can infer input_dim by len(axis=1) of dataset
                             lambda_tradeoff=lamdba_tradeoff, 
                             epochs=epochs,
                             patience=patience,
                             learning_rate=learning_rate,
                             adv_model_type=adv_model_type)
    

    model.fit(batched_dataset)

    X_val_array = np.array(X_val).astype(np.float32) 
    X_val_tensor = tf.convert_to_tensor(X_val_array, dtype=tf.float32)
    raw_preds = model.predict(X_val_tensor, raw_probabilities=True)

    preds = (raw_preds >= 0.11).astype(int)
    preds = preds.flatten()
    y_val = y_val.astype(int)

    fairness_df = X_val.copy()
    fairness_df['Coronary heart disease'] = y_val

    print('Finished Training')
    
    # Save all run results
    run_results_directory = f'{lamdba_tradeoff}_{epochs}_{learning_rate}_{patience}_{batch_size}_{adv_model_type}'
    run_results_directory = run_results_directory.replace('.', '')
    os.makedirs(os.path.join(RESULTS_DIRECTORY, run_results_directory), exist_ok=True)
    run_results_directory = os.path.join(RESULTS_DIRECTORY, run_results_directory)

    # save metrics just in case
    pd.DataFrame(model.results).to_csv(os.path.join(run_results_directory, 'epoch_results.csv'))

    print('Model assessment:')
    model_assessment_df = helpers.model_assessment(preds, y_val)
    print(model_assessment_df)
    model_assessment_df.to_csv(os.path.join(run_results_directory,'metrics_assessment.csv'))

    print('Model fairness metrics:')
    model_fairness_df = helpers.fairness_metrics(fairness_df, preds)
    print(model_fairness_df)
    model_fairness_df.to_csv(os.path.join(run_results_directory,'fairness_assessment.csv'))

    # plot metrics
    for metric in ['loss', 'accuracy', 'balanced_accuracy', 'main_model_loss', 'adv_model_loss']:
        plot_metric(x=model.results['epoch'], 
                    y=model.results[metric], 
                    x_label='epoch', 
                    y_label=metric, 
                    title=f'{metric} over epoch'.title().replace('_', ' '), 
                    metric=metric, 
                    plot_directory=run_results_directory)
        

    # F1 score
    thresholds = np.arange(0.01,1, 0.05)
    f1_scores = []
    for t in thresholds:
        preds = (raw_preds >= t).astype(int)
        f1 = f1_score(y_val, preds)
        f1_scores.append(f1)

    optimal_threshold = thresholds[f1_scores.index(max(f1_scores))]
    print(f"Optimal threshold for F1-Score: {optimal_threshold}")

    plot_metric(x=thresholds, 
                    y=f1_scores, 
                    x_label='Threshold', 
                    y_label='Score', 
                    title='F1-Score vs. Threshold', 
                    metric='f1_score', 
                    plot_directory=run_results_directory)