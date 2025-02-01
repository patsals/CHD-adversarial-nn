import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_decision_forests as tfdf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score,accuracy_score, precision_score, recall_score, f1_score, log_loss
import pandas as pd
import numpy as np
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
import time
from collections import deque

class AdversarialModel(keras.Model):
    def __init__(self, input_dim, sensitive_attr,lambda_tradeoff=0.1, GBT_retrain = 5, epochs = 100):
        super().__init__()

        # Initialize Attributes
        self.lambda_tradeoff = lambda_tradeoff  # Trade-off parameter for adversarial penalty
        self.sensitive_attr = sensitive_attr
        self.epochs = epochs
        self.GBT_retrain = GBT_retrain
    
        
   
        # Define the main neural network
        self.dense1 = Dense(32, activation='relu', input_dim = input_dim)
        self.dropout1 = Dropout(0.3)  # Added Dropout layer
        self.dense2 = Dense(16, activation='relu')
        self.output_layer = Dense(1, activation='sigmoid')  # Binary classification
    
        
        # Loss function and optimizer for Main Model
        self.loss_fn = keras.losses.BinaryCrossentropy()
        self.optimizer = keras.optimizers.Adam(learning_rate=0.001, loss= keras.losses.BinaryCrossentropy())

        # Adversarial model (Gradient Boosted Trees)
        self.adversarial_model = tfdf.keras.GradientBoostedTreesModel(n_estimators=100, learning_rate=0.1, task = tfdf.keras.Task.CLASSIFICATION)

    def call(self, inputs, train = False):
        """Forward pass"""
        x = self.dense1(inputs)
        x = self.dropout1(x, training = train)
        x = self.dense2(x)
        return self.output_layer(x)

    def fit(self, data):

        # Training Accuracy Metrics
        main_acc_metric = keras.metrics.BinaryCrossentropy()

        # Number of Batches
        num_batches = len(data)
       
        for epoch in range(self.epochs):
            
            # Epoch Progress Tracking
            start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            progbar = keras.utils.Progbar(target = num_batches)

            # Track epoch loss
            epoch_loss = 0

            # Adversarial Model is Trained at every K Epochs
            if epoch % self.GBT_retrain == 0:
                y_preds = []
                z_labels = []

                # Get Predictions For Most Recent Updated Main Model
                for step, (X_batch_train, _, z_batch_train) in enumerate(data):
                    y_pred = self(X_batch_train, train=False)  # Forward pass in mini-batch mode
                    y_preds.append(y_pred.numpy())
                    z_labels.append(z_batch_train.numpy())

                # Train the adversarial model on predictions vs sensitive attribute
                self.adversarial_model.fit(x=np.vstack(y_preds), y=np.vstack(z_labels))

                # Compute Adversarial Model Loss
                adversarial_model_loss  = self.adversarial_model.make_inspector().evaluation()[2]

           
            for step, (X_batch_train, y_batch_train) in enumerate(data):

                with tf.GradientTape() as tape:
                    # Forward pass
                    y_pred = self(X_batch_train, train=True)

                    # Compute Main Model Loss
                    main_model_loss = self.loss_fn(y_batch_train, y_pred)

                    # Compute Combined Loss
                    combined_loss = main_model_loss + (main_model_loss / adversarial_model_loss + 1e-7) - (self.lambda_tradeoff * adversarial_model_loss)

                # Compute gradients
                gradients = tape.gradient(combined_loss, self.trainable_weights)

                # Update weights
                self.optimizer.apply(zip(gradients, self.trainable_weights))

          
                 # Update training metric.
                main_acc_metric.update_state(y_batch_train, y_pred)

                # Track loss for epoch summary
                epoch_loss += combined_loss.numpy()


            # Update Progress Bar per batch
            progbar.update(step + 1, values=[("loss", float(combined_loss)), ("accuracy", float(main_acc_metric.result()))])
             
                    
        # Final calculations per epoch
        elapsed_time = time.time() - start_time
        time_per_step = elapsed_time / num_batches * 1e6  # Convert to microseconds
        final_accuracy = main_acc_metric.result().numpy()
        final_loss = epoch_loss / num_batches

        # Print final epoch stats
        print(f"\n{num_batches}/{num_batches} - {int(elapsed_time)}s {int(time_per_step)}us/step - accuracy: {final_accuracy:.4f} - loss: {final_loss:.4f}")

        # Reset Accuracy for Next Epoch
        main_acc_metric.reset_state()

    def predict(self, X_input, threshold = None, raw_probabilities = False):

    
        if threshold is None:
            threshold = 0.11

        pred_proba = super().predict(X_input)

        zpred_proba = self.adversarial_model.predict(pred_proba)

        if raw_probabilities == False:

            return pred_proba, zpred_proba
        
        else:
             
            binary_preds =  (pred_proba >= threshold).astype(int)
            binary_zpreds = (zpred_proba >= threshold).astype(int)

            return binary_preds, binary_zpreds
        

    # From https://keras.io/guides/custom_train_step_in_tensorflow/
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        loss = self.compute_loss(y=y, y_pred=y_pred)
        # Update the metrics.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}





        
                