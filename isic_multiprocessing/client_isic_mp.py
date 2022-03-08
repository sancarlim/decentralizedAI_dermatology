#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : client_isic_mp.py
# Modified   : 08.03.2022
# By         : Sandra Carrasco <sandra.carrasco@ai.se>

import numpy as np 
import os
from typing import List, Tuple, Dict

import torch 
import torch.nn as nn 
from torch import optim 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader 
from argparse import ArgumentParser 

import flwr as fl 
import utils
from utils import seed_everything  

import multiprocessing as mp
from utils import Net, seed_everything  , training_transforms, testing_transforms
import wandb 

import warnings

warnings.filterwarnings("ignore")
seed = 2022
seed_everything(seed)

EXCLUDE_LIST = [
    #"running",
    #"num_batches_tracked",
    #"bn",
]

class Client(fl.client.NumPyClient):
    """Flower client implementing melanoma classification using PyTorch."""

    def __init__(
        self, 
        device,
        nowandb,
        path,
    ) -> None:
        self.parameters = None 
        self.device = device
        self.nowandb = nowandb
        self.path = path

    def get_parameters(self) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return self.parameters

    def get_properties(self, config):
        return {}

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        self.parameters = parameters

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set parameters from the global model
        self.set_parameters(parameters)
        # Prepare multiprocess
        manager = mp.Manager()
        # We receive the results through a shared dictionary
        return_dict = manager.dict()
        # Create the process
        p = mp.Process(target=train, args=(args.model, parameters, return_dict, args.partition, args.num_partitions, 
                                                args.log_interval, args.epochs, 3, self.device, self.nowandb, self.path))
        # Start the process
        p.start() 
        # Wait for it to end
        p.join()
        # Close it
        try:
            p.close()
        except ValueError as e:
            print(f"Coudln't close the training process: {e}")
        # Get the return values
        new_parameters = return_dict["parameters"]
        data_size = return_dict["data_size"]
        train_loss = return_dict["train_loss"] 
        train_acc = return_dict["train_acc"] 
        val_loss = return_dict["val_loss"]
        val_acc = return_dict["val_acc"]
        # Del everything related to multiprocessing
        del (manager, return_dict, p)
        return new_parameters, data_size, {} 

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Prepare multiprocess
        manager = mp.Manager()
        # We receive the results through a shared dictionary
        return_dict = manager.dict()
        # Create the process
        p = mp.Process(target=utils.val_mp_server, args=(args.model, parameters, EXCLUDE_LIST, return_dict, self.device, self.path))
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
        auc = return_dict["auc_score"]
        num_examples = return_dict["num_examples"]
        # Del everything related to multiprocessing
        del (manager, return_dict, p)
        if not args.nowandb:
            wandb.log({f'Client{args.partition}/loss': loss, f'Client{args.partition}/accuracy': float(accuracy), f'Client{args.partition}/auc': float(auc)})
        return float(loss), num_examples["testset"], {"accuracy": float(accuracy), "auc": float(auc)}



def train(arch, parameters, return_dict, partition, num_partitions = 5, log_interval = 100, epochs = 10, es_patience = 3, device='cuda', nowandb=True, path='/workspace/melanoma_isic_dataset'):
    # Create model
    model = utils.load_model(arch, device)
    model.to(device)
    # Set model parameters, train model, return updated model parameters 
    if parameters is not None:
        utils.set_parameters(model, parameters, EXCLUDE_LIST)
    # Load data
    train_df, validation_df, num_examples = utils.load_isic_by_patient(partition, path)
    trainset = utils.CustomDataset(df = train_df, train = True, transforms = training_transforms) 
    valset = utils.CustomDataset(df = validation_df, train = True, transforms = testing_transforms )  
    train_loader = DataLoader(trainset, batch_size=32, num_workers=4, worker_init_fn=utils.seed_worker, shuffle=True) 
    val_loader = DataLoader(valset, batch_size=16, num_workers=4, worker_init_fn=utils.seed_worker, shuffle = False)     

    # Training model
    print('Starts training...')

    best_val = 0
    criterion = nn.BCEWithLogitsLoss()
    # Optimizer (gradient descent):
    optimizer = optim.Adam(model.parameters(), lr=0.0005) 
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=1, verbose=True, factor=0.2)

    patience = es_patience 

    for e in range(epochs):
        correct = 0
        running_loss = 0
        model.train()
        
        for i, (images, labels) in enumerate(train_loader):

            images, labels = images.to(device), labels.to(device)
                
            optimizer.zero_grad()
            
            output = model(images) 
            loss = criterion(output, labels.view(-1,1))  
            loss.backward()
            optimizer.step()
            
            # Training loss
            running_loss += loss.item()

            # Number of correct training predictions and training accuracy
            train_preds = torch.round(torch.sigmoid(output))
                
            correct += (train_preds.cpu() == labels.cpu().unsqueeze(1)).sum().item()
            
            #if i % log_interval == 0: 
            #    wandb.log({'training_loss': loss})
                            
        train_acc = correct / num_examples["trainset"]

        val_loss, val_auc_score, val_accuracy, val_f1 = utils.val(model, val_loader, criterion, partition, nowandb, device="cuda")
            
        print("Epoch: {}/{}.. ".format(e+1, epochs),
            "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
            "Training Accuracy: {:.3f}..".format(train_acc),
            "Validation Loss: {:.3f}.. ".format(val_loss/len(val_loader)),
            "Validation Accuracy: {:.3f}".format(val_accuracy),
            "Validation AUC Score: {:.3f}".format(val_auc_score),
            "Validation F1 Score: {:.3f}".format(val_f1))
        
        if not nowandb:
            wandb.log({f'Client{partition}/Training acc': train_acc, f'Client{partition}/training_loss': running_loss/len(train_loader),
                    f'Client{partition}/Validation AUC Score': val_auc_score, f'Client{partition}/Validation Acc': val_accuracy, f'Client{partition}/Validation Loss': val_loss})

        scheduler.step(val_auc_score)
                
        if val_auc_score > best_val:
            best_val = val_auc_score
            patience = es_patience  # Resetting patience since we have new best validation accuracy
            # model_path = os.path.join(f'./melanoma_fl_model_{best_val:.4f}.pth')
            # torch.save(model.state_dict(), model_path)  # Saving current best model
            # print(f'Saving model in {model_path}')
        else:
            patience -= 1
            if patience == 0:
                print('Early stopping. Best Val f1: {:.3f}'.format(best_val))
                break

    # Prepare return values
    return_dict["parameters"] = utils.get_parameters(model, EXCLUDE_LIST)
    return_dict["data_size"] = num_examples["trainset"] 
    return_dict["train_loss"] = running_loss/len(train_loader)
    return_dict["train_acc"] = train_acc
    return_dict["val_loss"] = val_loss/len(val_loader)
    return_dict["val_acc"] = val_accuracy

    del train_loader, val_loader, images                 

if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument("--model", type=str, default='efficientnet-b2')  
    parser.add_argument("--epochs", type=int, default='2')  
    parser.add_argument("--log_interval", type=int, default='100')  
    parser.add_argument("--num_partitions", type=int, default='10') 
    parser.add_argument("--partition", type=int, default='0')    
    parser.add_argument("--gpu", type=int, default='0')   
    parser.add_argument("--tags", type=str, default='Exp 5. FedAvg') 
    parser.add_argument("--path", type=str, default='/workspace/melanoma_isic_dataset') 
    parser.add_argument("--nowandb", action="store_true") 
    args = parser.parse_args()

    # Setting up GPU for processing or CPU if GPU isn't available
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if not args.nowandb:
        wandb.init(project="dai-healthcare" , entity='eyeforai', group='mp', tags=[args.tags], config={"model": args.model})
        wandb.config.update(args)

    # Set the start method for multiprocessing in case Python version is under 3.8.1
    mp.set_start_method("spawn", force=True)

    # Start client 
    fl.client.start_numpy_client("0.0.0.0:8080", Client(device, args.nowandb, args.path))

    