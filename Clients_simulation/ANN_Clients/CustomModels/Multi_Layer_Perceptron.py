import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten


class MultiLayerPerceptron:
    def __init__(self, epochs=1, lr=0.01):
        self.lr = lr
        self.epochs = epochs

        # Intitialising the model
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu',input_shape=(784,)))
        self.model.add(Dense(10, activation='softmax'))  # When multiclass classification
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        self.history = None
        print("Model Summary : ", self.model.summary())

    def fit(self, X_train, Y_train):
        # Ensure X_train and Y_train are numpy arrays
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)


        # Get the number of samples
        self.history = self.model.fit(X_train, Y_train, epochs=self.epochs, validation_split=0.2)

    def predict(self, X):
        y_prob =  self.model.predict(X)
        y_pred = y_prob.argmax(axis=1)
        return y_pred

    def get_loss_validation(self):
        metric = [self.history.history['loss'], self.history.history['val_loss'], self.history.history['accuracy'],self.history.history['val_accuracy']]

        return metric

    def update_parameters(self, parameters_dict):
        if len(parameters_dict) == 0:
            return
        self.model.set_weights(parameters_dict)

    def get_parameters(self):
        parameters = self.model.get_weights()
        return parameters

    def change_model_parameters(self, client_iter, client_learning_rate):
        self.n_iters = client_iter
        self.client_learning_rate = client_learning_rate
