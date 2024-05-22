import numpy as np
import aiohttp
import asyncio


class Server:
    def __init__(self,globals_parameters, max_round):
        self.globals_parameters = globals_parameters
        self.client_parameters = {}
        self.curr_round = 1
        self.max_round = 5
    
    def clear_client_parameters(self):
        self.client_parameters = {}

    def aggregate_weights_fedAvg_Neural(self):
        # Initialize a dictionary to hold the aggregated sums of vectors
        # print("Received Parameters : " , type(self.client_parameters[1][1][0]),
        #                                 len(self.client_parameters[1][1]),self.client_parameters[1][2][0][:5])

        # Count the number of clients
        num_clients = len(self.client_parameters)

        client_parameters = self.client_parameters
        for client in client_parameters:
            client_parameters[client] = [np.array(arr) for arr in client_parameters[client]]

        aggregated_sums = []
        for layer in range(len(client_parameters[1])):
            layer_dimension = client_parameters[1][layer].shape

            aggregated_layer = np.zeros(layer_dimension)

            for client in client_parameters:
                aggregated_layer += client_parameters[client][layer]
            aggregated_layer /= num_clients
            aggregated_sums.append(aggregated_layer.tolist())

        print("Aggregate Weights after FedAvg: ",type(aggregated_sums[0][1][0]),len(aggregated_sums[0][1]),aggregated_sums[2][0][:4])

        self.globals_parameters = aggregated_sums
        self.clear_client_parameters()

async def send_request(session, url, data, client_id,server):
    async with session.post(url, json=data) as response:

        print(f"Response from client {client_id}: {response.status}")
        response_data = await response.json()  # Return the JSON response
        client_id = response_data['client_id']
        client_parameters = response_data['parameters']
        evaluation = response_data['evaluation']
        server.client_parameters[client_id] = client_parameters




async def send_updated_parameters_to_clients(server,num_clients):
    updated_parameters = {"parameters" : server.globals_parameters,"round_num":server.curr_round}
    async with aiohttp.ClientSession() as session:
        tasks = []
        for client_id in range(1, num_clients + 1):
            client_url = f"http://localhost:500{client_id}/receive_parameters_and_run_model"
            task = asyncio.create_task(send_request(session, client_url, updated_parameters, client_id,server))
            tasks.append(task)
        await asyncio.gather(*tasks)  # Gather responses




# Modify the __main__ block accordingly to use the responses
if __name__ == '__main__':
    
    
    # Define the size of arrays
    m_size = 10
    c_size = 1
    num_clients = int(input("Number of clients: "))
    max_round = int(input("Number of rounds: "))


    # Initialize the global parameters dictionary
    global_parameters = {}


    server = Server(global_parameters,max_round)

    start_learning = input("Press 'Y' to start Federated Learning: ")
    if start_learning.upper() == 'Y':
        for i in range(1,max_round+1):
            print("-"*50)
            print(f"Round {i}")
            print("-"*50)
            loop = asyncio.get_event_loop()
            responses = loop.run_until_complete(send_updated_parameters_to_clients(server,num_clients))

            server.curr_round += 1


            # Aggregate
            server.aggregate_weights_fedAvg_Neural()
    else:
        print("Federated Learning not started.")