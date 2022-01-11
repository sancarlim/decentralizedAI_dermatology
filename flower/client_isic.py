from collections import OrderedDict
import numpy as np 
import os
from typing import List, Tuple, Dict

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

from melanoma_cnn_efficientnet import Net, Synth_Dataset, test, CustomDataset , confussion_matrix, seed_everything
from melanoma_cnn_efficientnet import training_transforms, testing_transforms, create_split
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
import time 

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score

import wandb
from client_synthetic import Client, train, val, test
import flwr as fl

seed = 1234
seed_everything(seed)

# Setting up GPU for processing or CPU if GPU isn't available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument("--data_path", type=str, default='/workspace/melanoma_isic_dataset')
    parser.add_argument("--model", type=str, default='efficientnet')
    parser.add_argument("--epochs", type=int, default='30')  
    parser.add_argument("--log_interval", type=int, default='500')  
    args = parser.parse_args()

    wandb.init(project="Sahlgrenska" , entity='sandracl72', config={"model": args.model})

    # ISIC Dataset

    df = pd.read_csv(os.path.join(args.data_path , 'train_concat.csv')) 
    train_img_dir = os.path.join(args.data_path ,'train/train/')
    
    df['image_name'] = [os.path.join(train_img_dir, df.iloc[index]['image_name'] + '.jpg') for index in range(len(df))]

    train_split, valid_split = train_test_split (df, stratify=df.target, test_size = 0.20, random_state=42) 
    train_df=pd.DataFrame(train_split)
    validation_df=pd.DataFrame(valid_split) 
    
    training_dataset = CustomDataset(df = train_df, train = True, transforms = training_transforms ) 
    testing_dataset = CustomDataset(df = validation_df, train = True, transforms = testing_transforms ) 
    
    num_examples = {"trainset" : len(training_dataset), "testset" : len(testing_dataset)} 

    train_loader = DataLoader(training_dataset, batch_size=32, num_workers=4, shuffle=True) 
    test_loader = DataLoader(testing_dataset, batch_size=16, shuffle = False)  

    arch = EfficientNet.from_pretrained('efficientnet-b2') if args.model=='efficientnet' else resnet50(pretrained=True)
    model = Net(arch=arch).to(device)


    # Start client
    client = Client(model, train_loader, test_loader, num_examples)
    fl.client.start_numpy_client("0.0.0.0:8080", client)

    