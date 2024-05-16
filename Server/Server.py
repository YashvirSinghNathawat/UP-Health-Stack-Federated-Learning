import numpy as np
import time
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

    def aggregate_weights_fedAvg(self):
        # Initialize a dictionary to hold the aggregated sums of vectors
        aggregated_sums = {}

        # Count the number of clients
        num_clients = len(self.client_parameters)

        # Iterate through each client's data
        for client_id, parameters in self.client_parameters.items():
            for key, vectors in parameters.items():
                if key not in aggregated_sums:
                    # print("Check : ",vectors)
                    aggregated_sums[key] = np.zeros_like(vectors)
                # Sum the vectors for the current key
                #print(key,aggregated_sums[key],type(vectors))
                aggregated_sums[key] += np.array(vectors)

        # Divide the aggregated sums by the number of clients and convert to list
        for key in aggregated_sums:
            aggregated_sums[key] /= num_clients
            aggregated_sums[key] = aggregated_sums[key].tolist()

        print("Aggregate Weights after FedAvg: ",aggregated_sums)

        self.globals_parameters = aggregated_sums
        self.clear_client_parameters()






async def send_request(session, url, data, client_id,server):
    async with session.post(url, json=data) as response:
        print(f"Response from client {client_id}: {response.status}")
        response_data = await response.json()  # Return the JSON response
        client_id = response_data['client_id']
        client_parameters = response_data['parameters']
        server.client_parameters[client_id] = client_parameters




async def send_updated_parameters_to_clients(server):
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
    m_size = 22
    c_size = 1
    num_clients = 2  # Assuming there are 5 clients

    # Initialize the global parameters dictionary
    global_parameters = {"m": [0 for i in range(m_size)], "c": [0 for i in range(c_size)]}


    max_round = 3
    server = Server(global_parameters,max_round)

    start_learning = input("Press 'Y' to start Federated Learning: ")
    if start_learning.upper() == 'Y':
        for i in range(1,max_round+1):
            print("-"*50)
            print(f"Round {i}")
            print("-"*50)
            loop = asyncio.get_event_loop()
            responses = loop.run_until_complete(send_updated_parameters_to_clients(server))

            print("\nClient Parameters : ",server.client_parameters,'\n')

            server.curr_round += 1
            # Aggregate 
            server.aggregate_weights_fedAvg()
    else:
        print("Federated Learning not started.")