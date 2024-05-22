import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten

class MultiLayerPerceptron:
    def __init__(self , epochs=1,lr=0.01):
        self.lr = lr
        self.epochs = epochs

        # Intitialising the model
        self.model = Sequential()
        self.model.add(Flatten(input_shape = (28,28)))
        self.model.add(Dense(64,activation = 'relu'))
        self.model.add(Dense(10,activation='softmax'))  # When multiclass classification

        self.model.compile(loss = 'sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

    def fit(self, X_train, Y_train):
        # Ensure X_train and Y_train are numpy arrays
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)


        # Get the number of samples
        history = self.model.fit(X_train,Y_train,epochs=self.epochs,validation_split=0.2)
        print(history)
        

    def predict(self, X):
        X = np.array(X)
        pred_test = np.dot(X,self.m) + self.c
        return pred_test
    
    def update_parameters(self,parameters_dict):
        if len(parameters_dict)==0:
            return

    
    def get_parameters(self):
        parameters = self.model.get_weights()
        return parameters
    
    def change_model_parameters(self,client_iter,client_learning_rate):
        self.n_iters = client_iter
        self.client_learning_rate = client_learning_rate
