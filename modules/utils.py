import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from umap import UMAP
import scanpy as sc
from matplotlib.lines import Line2D
import anndata
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.cluster import KMeans
import math

writer = SummaryWriter()
test_writer = SummaryWriter()


def contrastive_loss(y_true, y_pred, margin=1):
    square_pred = torch.square(y_pred)
    margin_square = torch.square(torch.clamp(margin - y_pred, min=0))
    loss = torch.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return loss

def euclid_dis(vects):
    x, y = vects
    x_flat = x.view(x.size(0), -1)  
    y_flat = y.view(y.size(0), -1)  
    sum_square = torch.sum(torch.square(x_flat - y_flat), axis=1, keepdim=True)
    return torch.sqrt(torch.maximum(sum_square, torch.tensor(torch.finfo(float).eps).to(sum_square.device)))

def calculate_accuracy(y_pred, y_true):
    pred_labels = (y_pred < 0.5).float()  
    correct = (pred_labels == y_true).float()  
    accuracy = correct.mean()  
    return accuracy

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0

    regularize_activation = 10
    regularize_entropy = 10
    l1_lambda = 0.1
    for (data_a, data_b), target in tqdm(dataloader):
        data_a, data_b = data_a.to(device), data_b.to(device)
        target  = target.to(device)
        optimizer.zero_grad()
        output = model(data_a, data_b)
        
        loss = contrastive_loss(target, output)

        #reg_loss = model.base_network.regularization_loss(regularize_activation, regularize_entropy)
        #loss = loss + l1_lambda * reg_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        writer.add_scalar("Loss/train_loss", loss.item(), epoch)
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def eval_model(model, dataloader, device, epoch):
    model.eval()
    total_accuracy = 0
    total_samples = 0

    with torch.no_grad():
        for (data_a, data_b), target in dataloader:
            data_a, data_b = data_a.to(device), data_b.to(device)
            target = target.to(device)
            output = model(data_a, data_b)
            
            loss = contrastive_loss(target, output)
            test_writer.add_scalar("Loss/test_loss", loss.item(), epoch)
            
            accuracy = calculate_accuracy(output, target)
            
            total_accuracy += accuracy.item() * data_a.size(0)
            total_samples += data_a.size(0)
    
    avg_accuracy = total_accuracy / total_samples
    return avg_accuracy


def get_data_splits(X, Y, split, n_splits=5, shuffle=True, random_state=None):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = list(kf.split(X))  
    train_index, test_index = splits[split]  
    
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    return X_train, X_test, Y_train, Y_test