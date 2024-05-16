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
class Client:
    """
    This client class is specifically for simulating SVM model
    """
    def __init__(self, client_id, data, target):
        self.client_id = client_id
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
            data, target, test_size=0.2, random_state=client_id)
        self.model = LinearRegression(n_iters=1000)  # Placeholder hyperparameters
        self.accuracy = None

    def train(self):
        # print("target_train uniques", np.unique(self.target_train))
        self.model.fit(self.data_train, self.target_train)

    def send_parameters_to_server(self, server_address, server_port):
        url = f"http://{server_address}:{server_port}/update_parameters"
        data = {
            'client_id': self.client_id,
            'parameters': self.model.get_parameters().tolist()
        }
        response = requests.post(url, json=data)
        if response.status_code != 200:
            print(f"Error sending parameters to server from Client {self.client_id}. Status code:", response.status_code)
            return
        print(f"Parameters sent to server from Client {self.client_id} successfully.")

    def receive_updated_parameters_from_server(self, server_address, server_port):
        url = f"http://{server_address}:{server_port}/get_updated_parameters"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error receiving updated parameters from server for Client {self.client_id}. Status code:", response.status_code)
            return
        updated_parameters = np.array(response.json()['parameters'])
        self.model.update_parameters(updated_parameters)
        print(f"Received updated parameters from server for Client {self.client_id}.")

    def client_execute(self, server_address, server_port):
        self.receive_updates(server_address, server_port)
        self.train()
        self.calculate_accuracy()
        self.send_weights(server_address, server_port)

    def update_model_with_parameters(self, parameters):
        self.model.update_parameters(np.array(parameters))
        self.train()
        print(f"Model updated for Client {self.client_id}.")

    def calculate_accuracy(self):
        # Evaluate accuracy on local data
        predictions = self.model.predict(self.data_test)
        self.accuracy = np.mean(predictions == self.target_test)
        return self.accuracy

@app.route('/receive_parameters_and_run_model', methods=['POST'])
def receive_parameters_and_run_model():
    parameters = request.json['parameters']
    print(f"Client {client_id}: Parameters = ",parameters)
    # client.update_model_with_parameters(parameters)
    return jsonify({"message": f"Model updated and trained for Client {client.client_id}."})

if __name__ == "__main__":
    """
        hyperparameters for clients model are defined in the client class, where the model is initialized for each client.
    """

    file_path = "client_data\client_1_data.csv"  # Adjust file path as needed
    data = pd.read_csv(file_path)
    X = data.drop(columns=['id', 'host_id', 'price'])
    y = data['price']
    client = Client(client_id, X, y)

    app.run(host='localhost', port=5001)  # Adjust port number as needed
