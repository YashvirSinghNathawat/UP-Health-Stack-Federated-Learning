from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from CustomModels.Multi_Layer_Perceptron import MultiLayerPerceptron
from sklearn.metrics import accuracy_score
import numpy as np

app = Flask(__name__)
client = None
client_id = 1
class Client:


    def __init__(self,client_id,data,target):
        self.client_id = client_id
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data, target, test_size=0.2, random_state=client_id)
        self.model = MultiLayerPerceptron()
        self.round_num = None

    def train(self):
        self.model.fit(self.X_train,self.y_train)

    def evaluate_metrics(self):
        y_pred = self.model.predict(self.X_test)

        test_acc = accuracy_score(self.y_test, y_pred)
        train_loss, val_loss ,train_acc,val_acc = self.model.get_loss_validation()
        print(f"Local Testing Accuracy for round {self.round_num}: {test_acc}")
        metrics_dict = {
            'test_acc': test_acc,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }
        return metrics_dict

    def update_model_with_parameters(self,parameters):
        # Check parameters Updated
        #print("Check kar before  : ", self.model.get_parameters()[2][0][:5])

        self.model.update_parameters(parameters)

        # Check parameters Updated
        #print("Check kar After  : ", self.model.get_parameters()[2][0][:5])

        self.train()
        evaluation = self.evaluate_metrics()

        #print(f"Model updated for Client {self.client_id}.")
        updated_parameters = self.model.get_parameters()
        #print("After Training new parameters : ",updated_parameters[2][0][:5])
        return [updated_parameters,evaluation]


def change_to_send_format(list_of_ndarray):
    list_of_lists = [arr.tolist() for arr in list_of_ndarray]
    return list_of_lists

def change_to_client_format(lists_of_list):
    list_of_ndarray = [np.array(par_list) for par_list in lists_of_list]
    return list_of_ndarray

@app.route('/receive_parameters_and_run_model',methods=['POST'])
def receive_parameters_and_run_model():
    request_data = request.json
    parameters = request_data['parameters']
    client.round_num = request_data['round_num']
    
    print('-'*50)
    print(f'Round {client.round_num}')
    print('-'*50)

    print("Received Parameters from Server : ", len(parameters))
    parameters = change_to_client_format(parameters)
    new_parameters,evaluation = client.update_model_with_parameters(parameters)

    new_parameters_serialized = change_to_send_format(new_parameters)

    return jsonify({"message": f"Model updated and trained for Client {client.client_id}.",
                "parameters": new_parameters_serialized,
                "client_id" : client_id,"evaluation" : evaluation })


if __name__ == "__main__":
    """
        hyperparameters for clients model are defined in the client class, where the model is initialized for each client.
    """

    file_path_X = f"client_data\client_{client_id}_data_X.csv"  # Adjust file path as needed
    X = pd.read_csv(file_path_X)

    file_path_y = f"client_data\client_{client_id}_data_Y.csv"  # Adjust file path as needed
    y = pd.read_csv(file_path_y)

    print(X.shape,y.shape)
    client = Client(client_id, X, y)

    app.run(host='localhost', port=5000 + client_id)  # Adjust port number as needed