from collections import OrderedDict
import numpy as np 
import os
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt  
from pathlib import Path
from PIL import Image 

import torch

from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet50
from torch.optim.lr_scheduler import ReduceLROnPlateau
from argparse import ArgumentParser 
import sys

sys.path.append('/workspace/stylegan2-ada-pytorch')

from melanoma_cnn_efficientnet import Net, CustomDataset,seed_everything
from melanoma_cnn_efficientnet import training_transforms, testing_transforms, create_split
import pandas as pd
from sklearn.model_selection import train_test_split 

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import wandb

import flwr as fl

seed = 1234
seed_everything(seed)

# Setting up GPU for processing or CPU if GPU isn't available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CifarClient(fl.client.NumPyClient):
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
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

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
        loss, auc, accuracy, f1 = val(self.model, self.testloader)
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
                wandb.log({'loss': loss})
                            
        train_acc = correct / len(training_dataset)

        val_loss, val_auc_score, val_accuracy, val_f1 = val(model, validate_loader, criterion)
            
        print("Epoch: {}/{}.. ".format(e+1, epochs),
            "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
            "Training Accuracy: {:.3f}..".format(train_acc),
            "Validation Loss: {:.3f}.. ".format(val_loss/len(validate_loader)),
            "Validation Accuracy: {:.3f}".format(val_accuracy),
            "Validation AUC Score: {:.3f}".format(val_auc_score),
            "Validation F1 Score: {:.3f}".format(val_f1))
            
        wandb.log({'Training acc': train_acc, 'training_loss': running_loss/len(train_loader),
                    'Validation AUC Score': val_auc_score, 'Validation Acc': val_accuracy, 'Validation Loss': val_loss})

        scheduler.step(val_auc_score)
                
        if val_auc_score > best_val:
            best_val = val_auc_score
            patience = es_patience  # Resetting patience since we have new best validation accuracy
            model_path = os.path.join(f'./melanoma_fl_model_{best_val:.4f}.pth')
            torch.save(model.state_dict(), model_path)  # Saving current best model
            print(f'Saving model in {model_path}')
        else:
            patience -= 1
            if patience == 0:
                print('Early stopping. Best Val f1: {:.3f}'.format(best_val))
                break

    del train_loader, validate_loader, images 

    return model
                
def val(model, validate_loader, criterion = nn.BCEWithLogitsLoss()):          
    model.eval()
    preds=[]            
    all_labels=[]
    criterion = nn.BCEWithLogitsLoss()
    # Turning off gradients for validation, saves memory and computations
    with torch.no_grad():
        
        val_loss = 0 
    
        for val_images, val_labels in validate_loader:
        
            val_images, val_labels = val_images.to(device), val_labels.to(device)
        
            val_output = model(val_images)
            val_loss += (criterion(val_output, val_labels.view(-1,1))).item() 
            val_pred = torch.sigmoid(val_output)
            
            preds.append(val_pred.cpu())
            all_labels.append(val_labels.cpu())
        pred=np.vstack(preds).ravel()
        pred2 = torch.tensor(pred)
        val_gt = np.concatenate(all_labels)
        val_gt2 = torch.tensor(val_gt)
            
        val_accuracy = accuracy_score(val_gt2, torch.round(pred2))
        val_auc_score = roc_auc_score(val_gt, pred)
        val_f1_score = f1_score(val_gt, np.round(pred))

        return val_loss, val_auc_score, val_accuracy, val_f1_score

def test(model, test_loader):
    test_preds=[]
    all_labels=[]
    with torch.no_grad():
        
        for _, (test_images, test_labels) in enumerate(test_loader):
            
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            
            test_output = model(test_images)
            test_pred = torch.sigmoid(test_output)
                
            test_preds.append(test_pred.cpu())
            all_labels.append(test_labels.cpu())
            
        test_pred=np.vstack(test_preds).ravel()
        test_pred2 = torch.tensor(test_pred)
        test_gt = np.concatenate(all_labels)
        test_gt2 = torch.tensor(test_gt)
        try:
            test_accuracy = accuracy_score(test_gt2.cpu(), torch.round(test_pred2))
            test_auc_score = roc_auc_score(test_gt, test_pred)
            test_f1_score = f1_score(test_gt, np.round(test_pred))
        except:
            test_auc_score = 0
            test_f1_score = 0
            pass

        wandb.log({"roc": wandb.plot.roc_curve(test_gt2, test_pred2)})    
        wandb.log({"pr": wandb.plot.pr_curve(test_gt2, test_pred2)})
        
        cm = wandb.plot.confusion_matrix(
            y_true=test_gt2,
            preds=test_pred2,
            class_names=["benign", "melanoma"])  
        wandb.log({"conf_mat": cm})

    print("Test Accuracy: {:.5f}, ROC_AUC_score: {:.5f}, F1 score: {:.4f}".format(test_accuracy, test_auc_score, test_f1_score))  

    return test_pred, test_gt, test_accuracy


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/workspace/generated-no-valset') 
    parser.add_argument("--model", type=str, default='efficientnet')
    parser.add_argument("--epochs", type=int, default='30')  
    parser.add_argument("--log_interval", type=int, default='500')   
    parser.add_argument("--n_imgs",  type=str, default="0,15", help='n benign, n melanoma K synthetic images to add to the real data')
    args = parser.parse_args()

    wandb.init(project="Sahlgrenska" , entity='sandracl72', config={"model": args.model})

    # Synthetic Dataset
    input_images = [str(f) for f in sorted(Path(args.data_path).rglob('*')) if os.path.isfile(f)]
    y = [0 if f.split('.jpg')[0][-1] == '0' else 1 for f in input_images]
    
    n_b, n_m = [int(i) for i in args.n_imgs.split(',') ] 
    train_id_list, val_id_list = create_split(args.data_path, n_b , n_m) 
    train_img = [input_images[int(i)] for i in train_id_list]
    train_gt = [y[int(i)] for i in train_id_list]
    train_img, test_img, train_gt, test_gt = train_test_split(input_images, y, stratify=y, test_size=0.2, random_state=3)
    synt_train_df = pd.DataFrame({'image_name': train_img, 'target': train_gt})
    synt_test_df = pd.DataFrame({'image_name': test_img, 'target': test_gt})
    
    training_dataset = CustomDataset(df = synt_train_df, train = True, transforms = training_transforms ) 
    testing_dataset = CustomDataset(df = synt_test_df, train = True, transforms = testing_transforms ) 
    
    num_examples = {"trainset" : len(training_dataset), "testset" : len(testing_dataset)} 

    train_loader = DataLoader(training_dataset, batch_size=32, shuffle=True) 
    test_loader = DataLoader(testing_dataset, batch_size=16, shuffle = False)  

    arch = EfficientNet.from_pretrained('efficientnet-b2') if args.model=='efficientnet' else resnet50(pretrained=True)
    model = Net(arch=arch).to(device)


    # Start client
    client = CifarClient(model, train_loader, test_loader, num_examples)
    fl.client.start_numpy_client("0.0.0.0:8080", client)

    