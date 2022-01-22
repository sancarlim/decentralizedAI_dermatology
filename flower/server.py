import sys
sys.path.append('/workspace/flower')
import src.py.flwr as fl 

if __name__ == "__main__":
    fl.server.start_server("0.0.0.0:8080", config={"num_rounds": 3}, strategy=fl.server.strategy.FedAvg())