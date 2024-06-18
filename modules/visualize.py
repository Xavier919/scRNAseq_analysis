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

def elbow_plot(adata):
    sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
    pca_variance_ratio = adata.uns['pca']['variance_ratio']
    fig, ax = plt.subplots()
    ax.bar(range(len(pca_variance_ratio)), pca_variance_ratio, alpha=0.6)
    ax.plot(range(len(pca_variance_ratio)), pca_variance_ratio, 'o', color='red')
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Variance ratio')
    ax.set_title('PCA variance ratio')
    plt.savefig('figures/elbow_plot.png')
    plt.show()

def dimension_heatmap(adata, n_components=15, n_cells=500):
    sc.tl.pca(adata, svd_solver='arpack')
    pca_df = pd.DataFrame(adata.obsm['X_pca'][:, :n_components], index=adata.obs_names)
    sampled_pca_df = pca_df.sample(n=n_cells)
    plt.figure(figsize=(8, 11))
    ax = sns.heatmap(sampled_pca_df.transpose(), cmap='viridis', cbar=True)
    ax.set_xlabel('Cells')
    ax.set_ylabel('Principal components')
    ax.set_title(f'Dimension heatmap')
    ax.set_xticks([])
    plt.savefig('figures/dimension_heatmap.png')
    plt.show()

def plot_umap(adata, sample_tags, legend_fontsize=5):
    for tag in sample_tags:
        adata_subset = adata[adata.obs['Sample_Tag'] == tag]
        sc.pl.umap(
            adata_subset,
            color=['annotated_cluster'],
            size=20,
            title=f'{tag}',
            save=f'umap_{tag}.png',
            legend_loc='on data',
            legend_fontsize=legend_fontsize
        )


def remove_numbers(cell_type):
    return re.sub(r'^\d+\s+', '', cell_type)

def assign_unique_cell_type_names(adata, cluster_key='leiden', cell_type_key='class_name'):
    cluster_annotations = {}
    cell_type_counter = defaultdict(int)
    
    for cluster in adata.obs[cluster_key].unique():
        most_common_cell_type = adata.obs[adata.obs[cluster_key] == cluster][cell_type_key].mode()[0]
        cleaned_cell_type = remove_numbers(most_common_cell_type)
        
        cell_type_counter[cleaned_cell_type] += 1
        unique_cell_type = f"{cleaned_cell_type}_{cell_type_counter[cleaned_cell_type]}"
        
        cluster_annotations[cluster] = unique_cell_type
    
    adata.obs['annotated_cluster'] = adata.obs[cluster_key].map(cluster_annotations)

def get_master_table(adata):
    merged_df = adata.obs[['Sample_Tag', 'annotated_cluster']]
    result_df = merged_df.groupby('annotated_cluster').size().reset_index(name='total_count')
    sample_tag_counts = merged_df.groupby(['annotated_cluster', 'Sample_Tag']).size().unstack(fill_value=0)
    sample_tag_counts.to_csv('figures/sample_tag_counts.csv')
    return sample_tag_counts

def get_ditto(sample_tag_counts):
    sample_tag_percentages = sample_tag_counts.div(sample_tag_counts.sum(axis=1), axis=0) * 100
    plt.figure(figsize=(14, 8))
    colormap = plt.get_cmap('viridis')
    for i, tag in enumerate(sample_tag_percentages.columns):
        plt.barh(sample_tag_percentages.index, sample_tag_percentages[tag], 
                 left=sample_tag_percentages[sample_tag_percentages.columns[:i]].sum(axis=1), 
                 label=tag, color=colormap(i / len(sample_tag_percentages.columns)))
    plt.xlabel('Percentage')
    plt.ylabel('Cluster')
    plt.title('Clusters sample tag %')
    plt.legend(title='Sample Tag', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('figures/ditto.png', bbox_inches='tight')
    plt.show()

def create_ditto_plot(adata, sample_tags, min_cell=50):
    """
    Creates a Ditto plot for the specified sample tags from an AnnData object.
    
    Parameters:
        adata (anndata.AnnData): The input AnnData object containing the observations.
        sample_tags (list of str): The sample tags for which to create the Ditto plot.
        min_cell (int): Minimum number of cells required to be considered a separate class. Classes with fewer cells will be grouped into 'Others'.
    """
    # Extract relevant columns and create DataFrame
    df = adata.obs[['Sample_Tag', 'class_name', 'annotated_cluster']]

    # Identify valid classes and group others based on all sample tags
    class_counts = df['class_name'].value_counts()
    valid_classes = class_counts[class_counts >= min_cell].index

    # Create color mapping for valid classes
    colormap = plt.get_cmap('viridis')  # Use 'tab20' for more distinct colors
    color_mapping = {class_name: colormap(i / len(valid_classes)) for i, class_name in enumerate(valid_classes)}

    # Filter for the specified sample tags
    df_filtered = df[df['Sample_Tag'].isin(sample_tags)]
    df_filtered['class_name'] = df_filtered['class_name'].apply(lambda x: x if x in valid_classes else 'Others')

    # Group by annotated_cluster and class_name, then count occurrences
    counts = df_filtered.groupby(['annotated_cluster', 'class_name']).size().unstack(fill_value=0).fillna(0)

    # Calculate percentages
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    # Plotting the Ditto plot
    plt.figure(figsize=(14, 8))

    # Plot bars and add cell count text
    for class_name in percentages.columns:
        color = color_mapping.get(class_name, 'grey')  # Default to grey if class_name not found
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

    # Labeling and title
    plt.xlabel('Percentage')
    plt.ylabel('Cluster')
    plt.title(f'Clusters cell type composition - {", ".join(sample_tags)}')
    
    # Create a legend with only the valid classes
    handles, labels = plt.gca().get_legend_handles_labels()
    valid_handles = [handles[i] for i, label in enumerate(labels)]
    valid_labels = [label for label in labels]
    plt.legend(valid_handles, valid_labels, title='Class Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(f'figures/ditto_by_class_name_{"_".join(sample_tags)}.png', bbox_inches='tight')
    plt.show()