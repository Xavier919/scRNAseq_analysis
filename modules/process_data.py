import pandas as pd
import numpy as np
import anndata
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from umap import UMAP
import scanpy as sc
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from scipy import sparse
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.cluster import KMeans
import math
import seaborn as sns
from matplotlib.colors import ListedColormap

mapping1 = {
 '01 IT-ET Glut': 1,
 '02 NP-CT-L6b Glut': 2,
 '03 OB-CR Glut': 3,
 '04 DG-IMN Glut': 4,
 '05 OB-IMN GABA': 5,
 '06 CTX-CGE GABA': 6,
 '07 CTX-MGE GABA': 7,
 '08 CNU-MGE GABA': 8,
 '09 CNU-LGE GABA': 9,
 '10 LSX GABA': 10,
 '11 CNU-HYa GABA': 11,
 '12 HY GABA': 12,
 '13 CNU-HYa Glut': 13,
 '14 HY Glut': 14,
 '15 HY Gnrh1 Glut': 15,
 '16 HY MM Glut': 16,
 '17 MH-LH Glut': 17,
 '18 TH Glut': 18,
 '19 MB Glut': 19,
 '20 MB GABA': 20,
 '21 MB Dopa': 21,
 '22 MB-HB Sero': 22,
 '23 P Glut': 23,
 '24 MY Glut': 24,
 '25 Pineal Glut': 25,
 '26 P GABA': 26,
 '27 MY GABA': 27,
 '28 CB GABA': 28,
 '29 CB Glut': 29,
 '30 Astro-Epen': 30,
 '31 OPC-Oligo': 31,
 '32 OEC': 32,
 '33 Vascular': 33,
 '34 Immune': 34
}

        
def filter_cells_by_gene_counts(adata, min_genes=150, max_genes=20000):
    nonzero_counts = (adata.X > 0).sum(axis=1)
    cell_mask = (nonzero_counts >= min_genes) & (nonzero_counts <= max_genes)
    filtered_adata = adata[cell_mask]
    return filtered_adata

def rm_high_mt(adata, threshold=0):
    mito_genes = [gene for gene in adata.var_names if gene.startswith('mt-')]
    total_counts = adata.X.sum(axis=1).A1 if isinstance(adata.X, np.matrix) else adata.X.sum(axis=1)
    mito_counts = adata[:, mito_genes].X.sum(axis=1).A1 if isinstance(adata.X, np.matrix) else adata[:, mito_genes].X.sum(axis=1)
    mito_percentage = mito_counts / total_counts
    cells_to_keep = mito_percentage <= threshold
    return adata[cells_to_keep, :]

def rm_low_exp(adata, threshold=0.05):
    nonzero_counts = np.array((adata.X != 0).sum(axis=0)).flatten()
    cell_count_threshold = (threshold / 100) * adata.shape[0]
    columns_to_keep = nonzero_counts >= cell_count_threshold
    return adata[:, columns_to_keep]

def elbow_plot(X):
    pca = PCA()
    pca.fit(X)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
        
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', markersize=3, linestyle='-')
    plt.axhline(y=0.90, color='r', linestyle='--')  
    plt.xlabel('Number of principal components')
    plt.ylabel('Cumulative explained variance')
    plt.title('Elbow plot for PCA')
    plt.grid(True)
    plt.show()

def scree_plot(X):
    pca = PCA()
    pca.fit(X)
    explained_variance_ratio = pca.explained_variance_ratio_
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center')
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance ratio')
    plt.title('Scree plot')
    plt.grid(True)
    plt.show()

def plot_umaps(X, Y, n_neighbors_list, color_mapping, min_count=100, classes_to_plot=None):
    markersize_scatter = 0.1
    markersize_legend = 10

    # Calculate the counts of each target
    target_counts = Counter(Y)

    # Filter unique targets based on the minimum count threshold
    filtered_targets = [target for target in np.unique(Y) if target_counts[target] >= min_count]

    # If classes_to_plot is provided, only keep those classes
    if classes_to_plot is not None:
        class_id = [int(x.split(' ')[0]) for x in filtered_targets]
        filtered_targets = [target for target in filtered_targets if int(target.split(' ')[0]) in classes_to_plot]

    fig, axes = plt.subplots(1, len(n_neighbors_list), figsize=(20, 6))

    for ax, n_neighbors in zip(axes, n_neighbors_list):
        umap_2d = UMAP(n_neighbors=n_neighbors, n_components=2, n_jobs=-1)
        X_umap = umap_2d.fit_transform(X)

        for target in filtered_targets:
            indices = np.where(Y == target)
            color = color_mapping.get(target, 'black')  # Default to black if the target is not found in the mapping
            ax.scatter(X_umap[indices, 0], X_umap[indices, 1], color=color, label=target, s=markersize_scatter)

        ax.set_title(f'n_neighbors = {n_neighbors}')
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping.get(target, 'black'), markersize=markersize_legend, label=target)
               for target in filtered_targets]

    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.grid(False)
    plt.show()

def plot_umap(X, Y, color_mapping, min_count=100, classes_to_plot=None):
    plt.figure(figsize=(6, 4))
    markersize_scatter = 0.1  
    markersize_legend = 10

    # Calculate the counts of each target
    target_counts = Counter(Y)
    
    # Filter unique targets based on the minimum count threshold
    filtered_targets = [target for target in set(Y) if target_counts[target] >= min_count]

    # If classes_to_plot is provided, only keep those classes
    if classes_to_plot is not None:
        class_id = [x.split(' ')[0] for x in filtered_targets]
        filtered_targets = [target for target in filtered_targets if target.split(' ')[0] in classes_to_plot]

    for target in filtered_targets:
        indices = np.where(Y == target)
        color = color_mapping.get(target, 'black')  # Default to black if the target is not found in the mapping
        plt.scatter(X[indices, 0], X[indices, 1], color=color, label=target, s=markersize_scatter)

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping.get(target, 'black'), markersize=markersize_legend, label=target)
               for target in filtered_targets]

    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)
    plt.grid(True)
    plt.show()



def gene_distribution(df, gene_id):
    filtered_df = df.loc[:, ~df.columns.str.startswith('mt-')]
    mean_values = filtered_df.mean()
    top_10_columns = mean_values.nlargest(10)
    print(top_10_columns)
    plt.figure(figsize=(10, 6))
    plt.hist(df[gene_id], bins=100, edgecolor='black')
    plt.title(f'Distribution of {gene_id} gene read count')
    plt.xlabel(f'{gene_id} read count')
    plt.ylabel('Number of cells')
    plt.show()


def visualize_umap(X, Y, mapping):
    plt.figure(figsize=(6, 4))
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
    plt.figure(figsize=(6, 4))
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
    
def plot_cluster_composition(cluster_composition):
    cluster_labels = list(cluster_composition.keys())
    all_sub_labels = sorted({sub_label for comp in cluster_composition.values() for sub_label in comp.keys()})
    composition_matrix = np.zeros((len(cluster_labels), len(all_sub_labels)))
    for i, cluster in enumerate(cluster_labels):
        for j, sub_label in enumerate(all_sub_labels):
            composition_matrix[i, j] = cluster_composition[cluster].get(sub_label, 0)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.jet(np.linspace(0, 1, len(all_sub_labels)))
    bottom = np.zeros(len(cluster_labels))
    for j, sub_label in enumerate(all_sub_labels):
        ax.bar(cluster_labels, composition_matrix[:, j], bottom=bottom, color=colors[j], label=sub_label)
        bottom += composition_matrix[:, j]
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Cell count')
    ax.set_title('Cluster composition')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=5)
    plt.xticks(cluster_labels)
    plt.grid(True)
    plt.show()

def get_pie_chart(cluster_composition, min_pct=1):
    num_charts = len(cluster_composition)
    num_cols = 3
    num_rows = math.ceil(num_charts / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 6*num_rows))
    
    # Ensure axes is always a list of Axes objects
    if num_rows * num_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    cluster_ids = list(cluster_composition.keys())
    
    for i, ax in enumerate(axes):
        if i < num_charts:
            cluster_id = cluster_ids[i]
            cluster_data = cluster_composition[cluster_id]
            
            # Exclude cluster ID -1
            if -1 in cluster_data:
                del cluster_data[-1]
            
            total = sum(cluster_data.values())
            sorted_data = sorted(cluster_data.items(), key=lambda item: item[1], reverse=True)
            top_labels, top_values, others_value = [], [], 0
            
            for label, value in sorted_data:
                percentage = (value / total) * 100
                if percentage < min_pct:
                    others_value += value
                else:
                    top_labels.append(label)
                    top_values.append(value)
            
            if others_value > 0:
                top_labels.append('Others')
                top_values.append(others_value)
            
            colors = plt.get_cmap('jet')(plt.Normalize(0, len(top_labels))(range(len(top_labels))))
            wedges, texts, autotexts = ax.pie(top_values, labels=top_labels, autopct='%1.1f%%', colors=colors, startangle=90, counterclock=False)
            plt.setp(autotexts, size=10, weight="bold", color="white")
            ax.axis('equal')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()



