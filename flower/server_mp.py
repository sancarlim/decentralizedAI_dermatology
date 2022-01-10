import flwr as fl
import flwr.server.strategy as strategy
import multiprocessing as mp
import argparse
from client_cifar import test, testloader, net

class FedAvgMp(strategy.FedAvg):
    """This class implements the FedAvg strategy for Multiprocessing context."""

    def configure_evaluate(self, rnd, parameters, client_manager):
        """Configure the next round of evaluation. Returns None since evaluation is made server side.
        You could comment this method if you want to keep the same behaviour as FedAvg."""
        return None

def get_eval_fn():
    """Get the evaluation function for server side.

    Returns
    -------
    evaluate
        The evaluation function
    """

    def evaluate(weights):
        """Evaluation function for server side."""

        # Prepare multiprocess
        manager = mp.Manager()
        # We receive the results through a shared dictionary
        return_dict = manager.dict()
        # Create the process
        p = mp.Process(target=test, args=(net, testloader, return_dict))
        # Start the process
        p.start()
        # Wait for it to end
        p.join()
        # Close it
        try:
            p.close()
        except ValueError as e:
            print(f"Coudln't close the evaluating process: {e}")
        # Get the return values
        loss = return_dict["loss"]
        accuracy = return_dict["accuracy"]
        # Del everything related to multiprocessing
        del (manager, return_dict, p)
        return float(loss), {"accuracy": float(accuracy)}

    return evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", type=int, default=3, help="Number of rounds for the federated training"
    )
    parser.add_argument(
        "-fc",
        type=int,
        default=2,
        help="Min fit clients, min number of clients to be sampled next round",
    )
    parser.add_argument(
        "-ac",
        type=int,
        default=2,
        help="Min available clients, min number of clients that need to connect to the server before training round can start",
    )
    args = parser.parse_args()
    rounds = int(args.r)
    fc = int(args.fc)
    ac = int(args.ac)

    # Set the start method for multiprocessing in case Python version is under 3.8.1
    mp.set_start_method("spawn")

    # Define the strategy
    strategy = FedAvgMp(
        fraction_fit=float(fc / ac),
        min_fit_clients=fc,
        min_available_clients=ac,
        eval_fn=get_eval_fn(),  
    )

    fl.server.start_server(
        "0.0.0.0:8080", config={"num_rounds": rounds}, strategy=strategy
    ) 