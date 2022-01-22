#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : client_isic.py
# Modified   : 22.01.2022
# By         : Sandra Carrasco <sandra.carrasco@ai.se>

from collections import OrderedDict
import numpy as np 
import os
from typing import List, Tuple, Dict

import torch 
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn 
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet50
from torch.optim.lr_scheduler import ReduceLROnPlateau
from argparse import ArgumentParser 

import flwr as fl 
import utils
from utils import Net, seed_everything  
 
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import wandb 

import warnings

warnings.filterwarnings("ignore")
seed = 1234
seed_everything(seed)

# Setting up GPU for processing or CPU if GPU isn't available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Client(fl.client.NumPyClient):
    """Flower client implementing melanoma classification using PyTorch."""

    def __init__(
        self,
        model: Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for name, val in self.model.state_dict().items()]# if 'bn' not in name]

    def get_properties(self, config):
        return {}

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        keys = [k for k in self.model.state_dict().keys()]# if 'bn' not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        utils.train(self.model, self.trainloader, self.testloader, self.num_examples, args.partition, args.log_interval, epochs=args.epochs, es_patience=3)
        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # WE DON'T EVALUATE OUR CLIENTS DECENTRALIZED
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, auc, accuracy, f1 = utils.val(self.model, self.testloader)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy), "auc": float(auc)}
                



if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument("--model", type=str, default='efficientnet') 
    parser.add_argument("--log_interval", type=int, default='100')  
    parser.add_argument("--epochs", type=int, default='2')  
    parser.add_argument("--num_partitions", type=int, default='10') 
    parser.add_argument("--partition", type=int, default='0')  
    args = parser.parse_args()

    wandb.init(project="dai-healthcare" , entity='eyeforai', group='FL' ,config={"model": args.model})
    wandb.config.update(args) 

    # Load model
    model = utils.load_model(args.model)

    # Load data
    trainset, testset, num_examples = utils.load_isic_data()
    # trainset, testset, num_examples = utils.load_partition(trainset, testset, num_examples, idx=args.partition, num_partitions=args.num_partitions)
    trainset, testset, num_examples = utils.load_experiment_partition(trainset, testset, num_examples, idx=args.partition)
    train_loader = DataLoader(trainset, batch_size=32, num_workers=4, shuffle=True) 
    test_loader = DataLoader(testset, batch_size=16, shuffle = False)   
    
    
    # Start client 
    client = Client(model, train_loader, test_loader, num_examples)
    fl.client.start_numpy_client("0.0.0.0:8080", client)

    