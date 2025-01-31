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

class AdversarialModel(keras.Model):
    def __init__(self, input_dim, sensitive_attr = str,lambda_tradeoff=0.1, metric = str, GBT_retrain = 5, epochs = 100, num_batches = 32):
        super().__init__()

        # Initialize Attributes
        self.lambda_tradeoff = lambda_tradeoff  # Trade-off parameter for adversarial penalty
        self.sensitive_attr = sensitive_attr
        self.epochs = epochs
        self.GBT_retrain = GBT_retrain
        self.metric = metric
        self.num_batches = num_batches
   
        # Define the main neural network
        self.dense1 = Dense(32, activation='relu', input_dim = input_dim)
        self.dropout1 = Dropout(0.3)  # Added Dropout layer
        self.dense2 = Dense(16, activation='relu')
        self.output_layer = Dense(1, activation='sigmoid')  # Binary classification
    
        
        # Loss function and optimizer for Main Model
        self.loss_fn = keras.losses.BinaryCrossentropy()
        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)

        # Adversarial model (Gradient Boosted Trees)
        self.adversarial_model = tfdf.keras.GradientBoostedTreesModel(n_estimators=100, learning_rate=0.1, task = tfdf.keras.Task.CLASSIFICATION)

    def call(self, inputs, train = False):
        """Forward pass"""
        x = self.dense1(inputs)
        x = self.dropout1(x, training = train)
        x = self.dense2(x)
        return self.output_layer(x)

    def fit(self, data):
        
        # Senstive Attribute
        z = data[self.sensitive_attr]

        # Loss Results
        main_model_train_loss_results = []
        combined_train_los_results = []

        for epoch in range(self.epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.BinaryAccuracy()

            # Epoch Progress Tracking
            start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            progbar = keras.utils.Progbar(target=self.num_batches)

            if epoch % self.GBT_retrain == 0:
                y_pred_full = []
                y_train_full = []

                for X_train, y_train in data:
                    y_pred_full.extend(self(X_train, train=False).numpy().flatten())
                    y_train_full.extend(y_train.numpy().flatten())

                # Train the adversarial model on predictions vs sensitive attribute
                self.adversarial_model.fit(x=y_pred_full, y=z)

                # Compute Adversarial Model Loss
                adversarial_model_loss  = self.adversarial_model.make_inspector().evaluation()[2]

            for batch_idx, (X_train, y_train) in enumerate(data):

                with tf.GradientTape() as tape:
                    # Forward pass
                    y_pred = self(X_train, train=True)

                    # Compute Main Model Loss
                    main_model_loss = self.loss_fn(y_train, y_pred)

                    if epoch % self.GBT_retrain == 0:
                        # Compute Combined Loss
                        combined_loss = main_model_loss + (main_model_loss / adversarial_model_loss) - (self.lambda_tradeoff * adversarial_model_loss)

                        # Compute gradients
                        trainable_vars = self.trainable_variables
                        gradients = tape.gradient(combined_loss, trainable_vars)

                        # Update weights
                        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

                         # Track Model Progress
                        epoch_loss_avg.update_state(combined_loss)
                        epoch_accuracy.update_state(y_train, y_pred)
                        
                    else:
                         # Compute gradients
                        trainable_vars = self.trainable_variables
                        gradients = tape.gradient(main_model_loss, trainable_vars)

                        # Update weights
                        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

                         # Track Model Progress
                        epoch_loss_avg.update_state(main_model_loss)
                        epoch_accuracy.update_state(y_train, y_pred)

            # End epoch
            main_model_train_loss_results.append(epoch_loss_avg.result())
            combined_train_los_results.append(epoch_accuracy.result())

            # Update Progress Bar
            progbar.update(1, values=[("accuracy", epoch_accuracy.result()), ("loss", epoch_loss_avg.result())])

            # Calculate elapsed time per step
            elapsed_time = time.time() - start_time
            time_per_epoch = elapsed_time / self.num_batches * 1e6  # Convert to microseconds

            print(f"{self.num_batches}/{self.num_batches} - {int(elapsed_time)}s {int(time_per_epoch)}us/step "
            f"- accuracy: {epoch_accuracy.result():.4f} - loss: {epoch_loss_avg.result():.4f}")

            if epoch > 0 and epoch % 100 == 0:
                self.save("/Saved_Models/Main_Model")
                self.adversarial_model.save_model("/Saved_Models/Adversarial_Model")

        # Save Final Neural Network Model
        self.save("/Saved_Models/Final_Model")

        return self


    def evaluate(self, X, y):

        loss_avg = tf.keras.metrics.Mean()
        accuracy_metric = tf.keras.metrics.BinaryAccuracy()



        
                