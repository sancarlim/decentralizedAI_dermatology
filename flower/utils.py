from collections import OrderedDict
import numpy as np 
import os
from typing import List, Tuple, Dict

import torch

from pathlib import Path
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet50 
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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(model = 'efficientnet'):
    arch = EfficientNet.from_pretrained('efficientnet-b2') if model=='efficientnet' else resnet50(pretrained=True)
    model = Net(arch=arch).to(DEVICE)

    return model


def load_isic_data():
    # ISIC Dataset

    df = pd.read_csv('/workspace/melanoma_isic_dataset/train_concat.csv')
    train_img_dir = '/workspace/melanoma_isic_dataset/train/train/'
    
    df['image_name'] = [os.path.join(train_img_dir, df.iloc[index]['image_name'] + '.jpg') for index in range(len(df))]

    train_split, valid_split = train_test_split (df, stratify=df.target, test_size = 0.20, random_state=42) 
    train_df=pd.DataFrame(train_split)
    validation_df=pd.DataFrame(valid_split) 
    
    training_dataset = CustomDataset(df = train_df, train = True, transforms = training_transforms ) 
    testing_dataset = CustomDataset(df = validation_df, train = True, transforms = testing_transforms ) 

    num_examples = {"trainset" : len(training_dataset), "testset" : len(testing_dataset)} 
    
    train_loader = DataLoader(training_dataset, batch_size=32, num_workers=4, shuffle=True) 
    test_loader = DataLoader(testing_dataset, batch_size=16, shuffle = False)  

    return train_loader, test_loader, num_examples


def load_synthetic_data(data_path, n_imgs):
    # Synthetic Dataset
    input_images = [str(f) for f in sorted(Path(data_path).rglob('*')) if os.path.isfile(f)]
    y = [0 if f.split('.jpg')[0][-1] == '0' else 1 for f in input_images]
    
    n_b, n_m = [int(i) for i in n_imgs.split(',') ] 
    train_id_list, val_id_list = create_split(data_path, n_b , n_m) 
    train_img = [input_images[int(i)] for i in train_id_list]
    train_gt = [y[int(i)] for i in train_id_list]
    test_img = [input_images[int(i)] for i in val_id_list]
    test_gt = [y[int(i)] for i in val_id_list]
    #train_img, test_img, train_gt, test_gt = train_test_split(input_images, y, stratify=y, test_size=0.2, random_state=3)
    synt_train_df = pd.DataFrame({'image_name': train_img, 'target': train_gt})
    synt_test_df = pd.DataFrame({'image_name': test_img, 'target': test_gt})
    
    training_dataset = CustomDataset(df = synt_train_df, train = True, transforms = training_transforms ) 
    testing_dataset = CustomDataset(df = synt_test_df, train = True, transforms = testing_transforms ) 
    
    num_examples = {"trainset" : len(training_dataset), "testset" : len(testing_dataset)} 

    train_loader = DataLoader(training_dataset, batch_size=32, shuffle=True) 
    test_loader = DataLoader(testing_dataset, batch_size=16, shuffle = False)  

    return train_loader, test_loader, num_examples


def val(model, validate_loader, criterion = nn.BCEWithLogitsLoss()):          
    model.eval()
    preds=[]            
    all_labels=[]
    criterion = nn.BCEWithLogitsLoss()
    # Turning off gradients for validation, saves memory and computations
    with torch.no_grad():
        
        val_loss = 0 
    
        for val_images, val_labels in validate_loader:
        
            val_images, val_labels = val_images.to(DEVICE), val_labels.to(DEVICE)
        
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
