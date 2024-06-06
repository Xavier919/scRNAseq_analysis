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


def visualize_umap(X, Y, mapping):
    plt.figure(figsize=(10, 8))
    unique_targets = np.unique(Y)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_targets)))
    markersize_scatter = 0.1  
    markersize_legend = 10
    for target, color in zip(unique_targets, colors):
        indices = np.where(Y == target)
        plt.scatter(X[indices, 0], X[indices, 1], color=color, label=mapping[target], s=markersize_scatter)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=markersize_legend, label=mapping[target])
               for target, color in zip(unique_targets, colors)]
    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.savefig('umap.png', bbox_inches='tight')
    plt.show()

def get_clustering(umap_result, kmeans_result):
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(kmeans_result)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        indices = np.where(kmeans_result == label)
        plt.scatter(umap_result[indices, 0], umap_result[indices, 1], color=color, label=f'Cluster {label}', s=0.1)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(True)
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Cluster {label}')
               for label, color in zip(unique_labels, colors)]
    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('kmeans.png', bbox_inches='tight')
    plt.show()
    
def get_cluster_composition(kmeans_result, Y):
    Y = np.array(Y)
    clusters = np.unique(kmeans_result)
    cluster_composition = {}
    for cluster in clusters:
        indices = np.where(kmeans_result == cluster)[0]
        cluster_labels = Y[indices]
        counter = Counter(cluster_labels)
        sorted_counter = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))
        cluster_composition[cluster] = {y: count for y, count in sorted_counter.items()}
    return cluster_composition
    
def plot_cluster_composition(cluster_composition, mapping):
    cluster_labels = list(cluster_composition.keys())
    all_sub_labels = sorted({sub_label for comp in cluster_composition.values() for sub_label in comp.keys()})
    composition_matrix = np.zeros((len(cluster_labels), len(all_sub_labels)))
    for i, cluster in enumerate(cluster_labels):
        for j, sub_label in enumerate(all_sub_labels):
            composition_matrix[i, j] = cluster_composition[cluster].get(sub_label, 0)
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.jet(np.linspace(0, 1, len(all_sub_labels)))
    bottom = np.zeros(len(cluster_labels))
    for j, sub_label in enumerate(all_sub_labels):
        ax.bar(cluster_labels, composition_matrix[:, j], bottom=bottom, color=colors[j], label=mapping[sub_label])
        bottom += composition_matrix[:, j]
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Cell count')
    ax.set_title('Cluster Composition')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(cluster_labels)
    plt.grid(True)
    plt.savefig('cluster_composition.png', bbox_inches='tight')
    plt.show()

def get_pie_chart(cluster_data, mapping, min_pct=1):
    total = sum(cluster_data.values())
    sorted_cluster_data = {k: v for k, v in sorted(cluster_data.items(), key=lambda item: item[1], reverse=True)}
    top_labels, top_values, others_value = [], [], 0
    for label, value in sorted_cluster_data.items():
        percentage = (value / total) * 100
        if percentage < min_pct:
            others_value += value
        else:
            top_labels.append(mapping[label])
            top_values.append(value)
    if others_value > 0:
        top_labels.append('Others')
        top_values.append(others_value)
    jet = plt.get_cmap('jet')
    norm = plt.Normalize(0, len(top_labels))
    colors = [jet(norm(i)) for i in range(len(top_labels))]
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        top_values, labels=top_labels, autopct='%1.1f%%', colors=colors,
        startangle=90, counterclock=False
    )
    plt.setp(autotexts, size=10, weight="bold", color="white")
    plt.axis('equal')  
    plt.savefig('pie_chart.png')
    plt.show()