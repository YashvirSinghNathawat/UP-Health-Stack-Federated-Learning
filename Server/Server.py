import numpy as np
import time
import aiohttp
import asyncio

weight_aggregated_time = time.time()
num_clients = 1  # Assuming there are 5 clients
num_rounds = 3

class Server:
    def __init__(self,globals_parameters, max_round):
        self.globals_parameters = globals_parameters
        self.update_queue = []
        self.curr_round = 1
        self.max_round = max_round

    def aggregate_weights(self, client_updates):
        print("Number of clients weights for aggregation:", len(client_updates))
        print(client_updates)
        # # Remove client_id from each clients
        # for client_data in client_updates:
        #     client_data.pop('client_id')
        #     client_data.pop('round')
    
        # for update in client_updates:
        #     for key in update.keys():
        #         avg_weights[key] += update[key]
        # if self.weights_dict is not None:
        #     print("%Change in weights after aggregating:", {key: np.mean(np.abs(avg_weights[key] - self.weights_dict[key]) / np.abs(self.weights_dict[key])) for key in avg_weights.keys()})
        # with weights_lock:
        #     self.weights_dict = {key: avg_weights[key] / len(client_updates) for key in avg_weights.keys()}








async def send_request(session, url, data, client_id):
    async with session.post(url, json=data) as response:
        print(f"Response from client {client_id}: {response.status}")
        return await response.json()  # Return the JSON response

async def send_updated_parameters_to_clients(server):
    updated_parameters = {"parameters" : server.globals_parameters}
    async with aiohttp.ClientSession() as session:
        tasks = []
        for client_id in range(1, num_clients + 1):
            client_url = f"http://localhost:500{client_id}/receive_parameters_and_run_model"
            task = asyncio.create_task(send_request(session, client_url, updated_parameters, client_id))
            tasks.append(task)
        responses = await asyncio.gather(*tasks)  # Gather responses
        return responses  # Return all responses




# Modify the __main__ block accordingly to use the responses
if __name__ == '__main__':
    
    globals_parameters = {"w":[0], "b":[0]}
    max_round = 3
    server = Server(globals_parameters,max_round)

    start_learning = input("Press 'Y' to start Federated Learning: ")
    if start_learning.upper() == 'Y':
        for i in range(1,max_round+1):
            print("-"*50)
            print(f"Round {i}")
            print("-"*50)
            loop = asyncio.get_event_loop()
            responses = loop.run_until_complete(send_updated_parameters_to_clients(server))
            print("Responses from clients:", responses)
    else:
        print("Federated Learning not started.")