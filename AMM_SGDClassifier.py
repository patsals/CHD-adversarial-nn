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


class AdversarialModel(keras.Model):
    def __init__(self, input_dim, lambda_tradeoff=0.1, epochs = 100, learning_rate=0.001):
        super().__init__()

        # Initialize Attributes
        self.lambda_tradeoff = lambda_tradeoff  # Trade-off parameter for adversarial penalty
        self.epochs = epochs
    
   
        # Define the main neural network
        self.dense1 = Dense(32, activation='relu', input_dim = input_dim)
        self.dropout1 = Dropout(0.3)  # Added Dropout layer
        self.dense2 = Dense(16, activation='relu')
        self.output_layer = Dense(1, activation='sigmoid')  # Binary classification
    
        
        # Metrics and optimizer for Main Model
        self.loss_fn = keras.losses.BinaryCrossentropy(name="loss")
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.main_acc_metric = keras.metrics.BinaryAccuracy(name="accuracy")

        # Adversarial model (SGD Classifier)
        self.adv_model = SGDClassifier(loss='log_loss', learning_rate="constant", eta0=0.01)

    def call(self, inputs, train = False):
        """Forward pass"""
        x = self.dense1(inputs)
        x = self.dropout1(x, training = train)
        x = self.dense2(x)
        return self.output_layer(x)
    
   
    def fit(self, data):

        # Number of Batches
        num_batches = len(data)
       
        for epoch in range(self.epochs):
            
            # Epoch Progress Tracking
            start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            progbar = keras.utils.Progbar(target = num_batches)

            # Track epoch loss
            epoch_loss = 0

           
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
                    adv_preds = self.adv_model.predict_proba(adv_input)[:, 1]

                    # Compute Adversarial Loss
                    adv_loss = log_loss(z_batch_train, adv_preds, labels=[0, 1])



                    # Compute Combined Loss
                    combined_loss = main_model_loss + (main_model_loss / (adv_loss + 1e-7)) - (self.lambda_tradeoff * adv_loss)

                # Compute gradients
                gradients = tape.gradient(combined_loss, self.trainable_weights)

                # Update weights
                self.optimizer.apply_gradients(list(zip(gradients, self.trainable_weights)))

          
                 # Update training metric.
                self.main_acc_metric.update_state(y_batch_train, y_pred)

                # Track loss for epoch summary
                epoch_loss += combined_loss.numpy()


            # Update Progress Bar per batch
            progbar.update(step + 1, values=[("loss", float(combined_loss)), ("accuracy", float(self.main_acc_metric.result()))])
             
                    
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
    
