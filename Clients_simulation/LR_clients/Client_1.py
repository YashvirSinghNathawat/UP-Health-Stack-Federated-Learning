import numpy as np
import random
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from CustomModels.Linear_Regression import LinearRegression
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify


app = Flask(__name__)
client_id = 1
client = None
class Client:
    """
    This client class is specifically for simulating SVM model
    """
    def __init__(self, client_id, data, target):
        self.client_id = client_id
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
            data, target, test_size=0.2, random_state=client_id)
        self.model = LinearRegression(n_iters=1)  # Placeholder hyperparameters
        self.accuracy = None
        self.round_num = None

    def train(self):
        # print("target_train uniques", np.unique(self.target_train))
        self.model.fit(self.data_train, self.target_train)

    def calculate_accuracy(self):
        # Evaluate accuracy on local data
        predictions = self.model.predict(self.data_test)
        print("Predictions : ",predictions.shape)
        self.accuracy = np.mean(predictions == self.target_test)
        print(f"Accuracy for round {self.round_num}: {self.accuracy}")

    def update_model_with_parameters(self, parameters):
        self.model.update_parameters(parameters)
        self.train()
        self.calculate_accuracy()
        print(f"Model updated for Client {self.client_id}.")
        return self.model.get_parameters()


@app.route('/receive_parameters_and_run_model', methods=['POST'])
def receive_parameters_and_run_model():
    request_data = request.json
    parameters = request_data['parameters']
    client.round_num = request_data['round_num']
    #print(f"Client {client_id}: Parameters = ",parameters)
    new_parameters = client.update_model_with_parameters(parameters)
    print("New Parameters = ",new_parameters)
    return jsonify({"message": f"Model updated and trained for Client {client.client_id}.",
                "parameters": new_parameters,
                "client_id" : client_id})

if __name__ == "__main__":
    """
        hyperparameters for clients model are defined in the client class, where the model is initialized for each client.
    """

    file_path = "client_data\client_1_data.csv"  # Adjust file path as needed
    data = pd.read_csv(file_path)
    X = data.drop(columns=['id', 'price','host_id'])
    y = data['price']
    client = Client(client_id, X, y)

    app.run(host='localhost', port=5001)  # Adjust port number as needed
