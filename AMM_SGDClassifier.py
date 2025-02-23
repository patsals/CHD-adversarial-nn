import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
import time
import helpers


class AdversarialModel(keras.Model):
    def __init__(self, input_dim, lambda_tradeoff=0.1, epochs = 100, learning_rate=0.001, patience=10, adv_model_type='logistic_regression'):
        super().__init__()

        # Initialize Attributes
        self.lambda_tradeoff = lambda_tradeoff  # Trade-off parameter for adversarial penalty
        self.epochs = epochs
        self.patience = patience
    
   
        # Define the main neural network
        self.dense1 = Dense(32, activation='relu', input_dim = input_dim)
        self.dropout1 = Dropout(0.3)  # Added Dropout layer
        self.dense2 = Dense(16, activation='relu')
        self.output_layer = Dense(1, activation='sigmoid')  # Binary classification
        # Metrics and optimizer for Main Model
        self.loss_fn = keras.losses.BinaryCrossentropy(name="loss")
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.main_acc_metric = keras.metrics.BinaryAccuracy(name="accuracy")
        self.balanced_acc = BalancedAccuracy()

        # Adversarial model (SGD Classifier)
        if adv_model_type == 'perceptron':
            loss_type = 'perceptron'
        elif adv_model_type == 'svm':
            loss_type = 'hinge'
        else:
            loss_type = 'log_loss'

        self.adv_model = SGDClassifier(loss=loss_type, learning_rate="constant", eta0=0.01, n_jobs=-1)

        # Store test results
        self.results = {
            'epoch': [],
            'accuracy': [],
            'balanced_accuracy': [],
            'loss': [],
            'main_model_loss': [],
            'adv_model_loss': [],
            'demographic_parity_difference': [],
            'equal_opportunity_difference': [],
            'disparate_impact': []
        }


    def call(self, inputs, train = False):
        """Forward pass"""
        x = self.dense1(inputs)
        x = self.dropout1(x, training = train)
        x = self.dense2(x)
        return self.output_layer(x)
    
   
    def fit(self, data, X_train, y_train):

        # Number of Batches
        num_batches = len(data)
       
        for epoch in range(self.epochs):
            
            # Epoch Progress Tracking
            start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            progbar = keras.utils.Progbar(target = num_batches)

            # Track epoch loss
            epoch_loss = 0
            total_main_model_loss = 0
            total_adv_loss = 0
           
            for step, (X_batch_train,z_batch_train, y_batch_train) in enumerate(data):

                with tf.GradientTape() as tape:
                    #Conver z_batch_train to Numpy Array for Adversarial Model
                    z_batch_train = z_batch_train.numpy().flatten()
                    # Forward pass
                    y_pred = self(X_batch_train, train=True)
                    adv_input = tf.stop_gradient(y_pred).numpy().reshape(-1, 1)

                    # Compute Main Model Loss
                    main_model_loss = self.loss_fn(y_batch_train, y_pred)

                    # Compute Adversarial Predictions
                    self.adv_model.partial_fit(adv_input , z_batch_train, classes=np.array([0, 1]))

                    # Handle cases where predict_proba is unavailable
                    if self.adv_model.loss in ['hinge', 'perceptron']:
                        raw_scores = self.adv_model.decision_function(adv_input)  # Get raw margin scores
                        adv_preds = tf.sigmoid(raw_scores)  
                    else:
                        adv_preds = self.adv_model.predict_proba(adv_input)[:, 1] 
                    self.adv_model.warm_start=True

                    # Compute Adversarial Loss
                    adv_loss = log_loss(z_batch_train, adv_preds, labels=[0, 1])

                    # Compute Combined Loss
                    combined_loss = main_model_loss + (main_model_loss / (adv_loss)) - (self.lambda_tradeoff * adv_loss)

                # Compute gradients
                gradients = tape.gradient(combined_loss, self.trainable_weights)

                # Update weights
                self.optimizer.apply_gradients(list(zip(gradients, self.trainable_weights)))

          
                 # Update training metric.
                self.main_acc_metric.update_state(y_batch_train, y_pred)
                self.balanced_acc.update_state(y_batch_train, y_pred)

                # Track loss for epoch summary
                epoch_loss += combined_loss.numpy()
                total_main_model_loss += main_model_loss.numpy()
                total_adv_loss += adv_loss

            # calculate balanced accuracy and fairness metrics
            X_train_array = np.array(X_train).astype(np.float32) 
            X_train_tensor = tf.convert_to_tensor(X_train_array, dtype=tf.float32)
            raw_preds = self.predict(X_train_tensor, raw_probabilities=True)
            y_preds = (raw_preds >= 0.11).astype(int).flatten()
            # y_train = y_train.astype(int)

            balanced_acc = balanced_accuracy(y_train.astype(int), raw_preds.flatten())

            # X_train['Coronary heart disease'] = y_train
            fairness_metrics_df = helpers.fairness_metrics(X_train.assign(**{'Coronary heart disease':y_train}), y_preds)
            demographic_parity_difference = fairness_metrics_df[fairness_metrics_df['Metric'] == \
                                                           'Demographic Parity Difference']['Value'].max()
            equal_opportunity_difference = fairness_metrics_df[fairness_metrics_df['Metric'] == \
                                                                'Equal Opportunity Difference']['Value'].max()
            disparate_impact = fairness_metrics_df[fairness_metrics_df['Metric'] == \
                                                                'Disparate Impact']['Value'].max()

            # Update Progress Bar per batch
            progbar.update(step + 1, values=[("total loss", float(epoch_loss)), 
                                             ("main model loss", float(total_main_model_loss)),
                                             ("adv loss", float(total_adv_loss)),
                                             ("accuracy", float(self.main_acc_metric.result())),
                                            #  ("balanced accuracy", self.balanced_acc.result())
                                            ("balanced accuracy", balanced_acc)
                                             ])
            
            # Store all epoch metrics in results
            self.results['epoch'].append(epoch)
            self.results['accuracy'].append(float(self.main_acc_metric.result()))
            self.results['loss'].append(float(epoch_loss))
            self.results['main_model_loss'].append(float(total_main_model_loss))
            self.results['adv_model_loss'].append(float(total_adv_loss))
            self.results['balanced_accuracy'].append(float(balanced_acc))
            self.results['demographic_parity_difference'].append(float(demographic_parity_difference))
            self.results['equal_opportunity_difference'].append(float(equal_opportunity_difference))
            self.results['disparate_impact'].append(float(disparate_impact))


            # Evaluate patience
            if epoch > self.patience and max(self.results['main_model_loss'][-(self.patience+1):-1]) < self.results['main_model_loss'][-1]:
                print('Reached patience level, ending training...')
                return 
             
                    
        # Final calculations per epoch
        elapsed_time = time.time() - start_time
        time_per_step = elapsed_time / num_batches * 1e6  # Convert to microseconds
        final_accuracy = self.main_acc_metric.result().numpy()
        final_loss = epoch_loss / num_batches

        # Print final epoch stats
        print(f"\n{num_batches}/{num_batches} - {int(elapsed_time)}s {int(time_per_step)}us/step - accuracy: {final_accuracy:.4f} - loss: {final_loss:.4f}")

        # Reset Accuracy for Next Epoch
        self.main_acc_metric.reset_state()

    def predict(self, X_input, threshold = None, raw_probabilities = None):

        if threshold is None:
            threshold = 0.11

        if raw_probabilities is None:
            raw_probabilities = False

        pred_proba = super().predict(X_input)

        if raw_probabilities == True:
            return pred_proba
        else:
            binary_preds =  (pred_proba >= threshold).astype(int)
            return binary_preds
    
def balanced_accuracy(y_true, y_pred, threshold=0.5):
    """
    Compute balanced accuracy given true labels and predicted probabilities.
    
    Args:
    - y_true (numpy array): Ground truth labels (0 or 1).
    - y_pred (numpy array): Predicted probabilities (0-1 range).
    - threshold (float): Probability threshold to classify as 1.
    
    Returns:
    - Balanced Accuracy (float)
    """
    # Convert probabilities to binary predictions
    y_pred = (y_pred > threshold).astype(int)

    # True Positives, False Negatives, True Negatives, False Positives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    # Sensitivity (Recall)
    sensitivity = tp / (tp + fn + 1e-7)  # Avoid division by zero
    specificity = tn / (tn + fp + 1e-7)

    # Balanced Accuracy
    return (sensitivity + specificity) / 2


class BalancedAccuracy(keras.metrics.Metric):
    def __init__(self, name="balanced_accuracy", **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")
        self.true_negatives = self.add_weight(name="tn", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Ensure y_true and y_pred are float32 tensors
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Convert probabilities to binary labels

        # Compute confusion matrix elements
        tp = tf.reduce_sum(tf.cast(tf.math.equal(y_true * y_pred, 1), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.math.equal(y_true - y_pred, 1), tf.float32))
        tn = tf.reduce_sum(tf.cast(tf.math.equal(y_true + y_pred, 0), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.math.equal(y_pred - y_true, 1), tf.float32))

        # Update state variables
        self.true_positives.assign_add(tp)
        self.false_negatives.assign_add(fn)
        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)

    def result(self):
        # Compute sensitivity (recall) and specificity
        sensitivity = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        specificity = self.true_negatives / (self.true_negatives + self.false_positives + tf.keras.backend.epsilon())

        # Balanced Accuracy = (Sensitivity + Specificity) / 2
        return (sensitivity + specificity) / 2

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_negatives.assign(0.0)
        self.true_negatives.assign(0.0)
        self.false_positives.assign(0.0)
