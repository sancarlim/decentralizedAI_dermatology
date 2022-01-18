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
        train(self.model, self.trainloader, self.testloader, epochs=1, es_patience=3)
        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, auc, accuracy, f1 = utils.val(self.model, self.testloader)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy), "auc": float(auc)}


def train(model, train_loader, validate_loader,  epochs = 10, es_patience = 3):
    # Training model
    print('Starts training...')

    best_val = 0
    criterion = nn.BCEWithLogitsLoss()
    # Optimizer (gradient descent):
    optimizer = optim.Adam(model.parameters(), lr=0.0005) 
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=1, verbose=True, factor=0.2)

    patience = es_patience 
    model.to(device)

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
            
            if i % args.log_interval == 0: 
                wandb.log({'training_loss': loss})
                            
        train_acc = correct / num_examples["trainset"]

        val_loss, val_auc_score, val_accuracy, val_f1 = utils.val(model, validate_loader, criterion)
            
        print("Epoch: {}/{}.. ".format(e+1, epochs),
            "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
            "Training Accuracy: {:.3f}..".format(train_acc),
            "Validation Loss: {:.3f}.. ".format(val_loss/len(validate_loader)),
            "Validation Accuracy: {:.3f}".format(val_accuracy),
            "Validation AUC Score: {:.3f}".format(val_auc_score),
            "Validation F1 Score: {:.3f}".format(val_f1))
            
        wandb.log({'Client/Training acc': train_acc, 'Client/training_loss': running_loss/len(train_loader),
                    'Client/Validation AUC Score': val_auc_score, 'Client/Validation Acc': val_accuracy, 'Client/Validation Loss': val_loss})

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

    del train_loader, validate_loader, images 

    return model
                


if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument("--model", type=str, default='efficientnet') 
    parser.add_argument("--log_interval", type=int, default='100')  
    parser.add_argument("--partition", type=int, default='0')  
    args = parser.parse_args()

    wandb.init(project="dai-healthcare" , entity='eyeforai', config={"model": args.model})

    # Load model
    model = utils.load_model(args.model)

    # Load data
    trainset, testset, num_examples = utils.load_isic_data()
    trainset, testset, num_examples = utils.load_partition(trainset, testset, num_examples, idx=args.partition)
    train_loader = DataLoader(trainset, batch_size=32, num_workers=4, shuffle=True) 
    test_loader = DataLoader(testset, batch_size=16, shuffle = False)  
    
    # Start client
    client = Client(model, train_loader, test_loader, num_examples)
    fl.client.start_numpy_client("0.0.0.0:8080", client)
