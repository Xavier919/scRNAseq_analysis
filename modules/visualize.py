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
import re
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def plot_top_n_distr(adata, top_n=5, save_path=None):
    non_mt_genes = adata.var[~adata.var.index.str.startswith('mt-')]
    top_genes = non_mt_genes['Raw_Reads'].sort_values(ascending=False).head(5).index
    top_genes_data = adata[:, top_genes].X
    top_genes_df = pd.DataFrame(top_genes_data.toarray(), columns=top_genes, index=adata.obs_names)
    for gene in top_genes:
        plt.figure(figsize=(8, 6))
        sns.histplot(top_genes_df[gene], bins=50, kde=True, label=gene)
        plt.xlabel('Number of cells')
        plt.xlim(0, 500)
        plt.ylabel('Number of reads')
        plt.title(f'{gene} - Number of reads per cell')
        plt.legend(title='Genes')
        if save_path:
            plt.savefig(f'{save_path}_{gene}.png')
        plt.show()

def elbow_plot(adata, save_path=None):
    sc.tl.pca(adata, svd_solver='arpack', n_comps=500)
    pca_variance_ratio = adata.uns['pca']['variance_ratio']
    fig, ax = plt.subplots()
    ax.bar(range(len(pca_variance_ratio)), pca_variance_ratio, alpha=0.6)
    ax.plot(range(len(pca_variance_ratio)), pca_variance_ratio, 'o', color='red')
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Variance ratio')
    ax.set_title('PCA variance ratio')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def dimension_heatmap(adata, n_components=15, n_cells=500, save_path=None):
    sc.tl.pca(adata, svd_solver='arpack')
    pca_df = pd.DataFrame(adata.obsm['X_pca'][:, :n_components], index=adata.obs_names)
    sampled_pca_df = pca_df.sample(n=n_cells)
    plt.figure(figsize=(8, 11))
    ax = sns.heatmap(sampled_pca_df.transpose(), cmap='viridis', cbar=True)
    ax.set_xlabel('Cells')
    ax.set_ylabel('Principal components')
    ax.set_title(f'Dimension heatmap')
    ax.set_xticks([])
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_umap(adata, cluster_type='cluster_class_name', legend_fontsize=5, save_path=None):
    sample_tags = adata.obs['Sample_Tag'].unique()
    for tag in sample_tags:
        adata_subset = adata[adata.obs['Sample_Tag'] == tag]
        sc.pl.umap(
            adata_subset,
            color=[cluster_type],
            size=20,
            title=f'{tag}',
            save=f'{save_path}_{tag}.png',
            legend_loc='on data',
            legend_fontsize=legend_fontsize
        )

def remove_numbers(cell_type):
    return re.sub(r'^\d+\s+', '', cell_type)

def assign_unique_cell_type_names(adata, cluster_key='leiden', cluster_types=['class_name', 'subclass_name']):
    for cluster_type in cluster_types:
        cluster_annotations = {}
        cell_type_counter = defaultdict(int)
        
        for cluster in adata.obs[cluster_key].unique():
            most_common_cell_type = adata.obs[adata.obs[cluster_key] == cluster][cluster_type].mode()[0]
            cleaned_cell_type = remove_numbers(most_common_cell_type)
            
            cell_type_counter[cleaned_cell_type] += 1
            unique_cell_type = f"{cleaned_cell_type}_{cell_type_counter[cleaned_cell_type]}"
            
            cluster_annotations[cluster] = unique_cell_type
        
        adata.obs[f'cluster_{cluster_type}'] = adata.obs[cluster_key].map(cluster_annotations)
        adata.obs[f'cluster_{cluster_type}'] = adata.obs[f'cluster_{cluster_type}'].astype(str)
    
    adata.obs[cluster_key] = adata.obs[cluster_key].astype(str)

def get_master_table(adata, cluster_type='cluster_class_name', save_path=None):
    merged_df = adata.obs[['Sample_Tag', cluster_type]]
    result_df = merged_df.groupby(cluster_type).size().reset_index(name='total_count')
    sample_tag_counts = merged_df.groupby([cluster_type, 'Sample_Tag']).size().unstack(fill_value=0)
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(save_path, f)
    return sample_tag_counts

def create_ditto_plot(adata, sample_tags, class_level, cluster_type, min_cell=50, save_path=None):
    df = adata.obs[['Sample_Tag', class_level, cluster_type]]
    class_counts = df[class_level].value_counts()
    valid_classes = class_counts[class_counts >= min_cell].index
    colormap = plt.get_cmap('tab20')  # Use 'tab20' for more distinct colors
    color_mapping = {class_name: colormap(i / len(valid_classes)) for i, class_name in enumerate(valid_classes)}
    df_filtered = df[df['Sample_Tag'].isin(sample_tags)]
    df_filtered[class_level] = df_filtered[class_level].apply(lambda x: x if x in valid_classes else 'Others')
    counts = df_filtered.groupby([cluster_type, class_level]).size().unstack(fill_value=0).fillna(0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    plt.figure(figsize=(14, 8))
    for class_name in percentages.columns:
        color = color_mapping.get(class_name, 'grey')  
        bars = plt.barh(percentages.index, percentages[class_name], 
                        left=percentages[percentages.columns[:percentages.columns.get_loc(class_name)]].sum(axis=1), 
                        label=class_name, color=color)
        for bar in bars:
            width = bar.get_width()
            cluster = bar.get_y()
            if cluster in counts.index and class_name in counts.columns:
                cell_count = int(counts.at[cluster, class_name])
                plt.text(width, bar.get_y() + bar.get_height()/2, 
                         f'{cell_count}', 
                         va='center', ha='left')

    plt.xlabel('Percentage')
    plt.ylabel('Cluster')
    plt.title(f'Clusters cell type composition - {", ".join(sample_tags)}')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    valid_handles = [handles[i] for i, label in enumerate(labels)]
    valid_labels = [label for label in labels]
    plt.legend(valid_handles, valid_labels, title='Class Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()