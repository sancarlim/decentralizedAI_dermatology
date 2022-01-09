import flwr as fl
import flwr.server.strategy as strategy

if __name__ == "__main__":
    fl.server.start_server("0.0.0.0:8080", config={"num_rounds": 3}, strategy=strategy.FedAvg())