import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import scanpy as sc
import seaborn as sns
import scipy.stats as stats
import re
from collections import defaultdict
import pickle
from collections import Counter
import matplotlib.cm as cm

def pie_chart_condition(list_):
    string_counts = Counter(list_)
    labels = list(string_counts.keys())
    sizes = list(string_counts.values())
    fig, ax = plt.subplots(figsize=(4, 4))
    colors = cm.viridis([i / len(labels) for i in range(len(labels))])
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.axis('equal')  
    plt.show()

def plot_top_genes_qq(adata, top_n=3, save_path=None):
    """
    Plot QQ plots for the top N genes with the highest raw reads, excluding mitochondrial genes.
    
    Parameters:
    adata (AnnData): The AnnData object containing gene expression data.
    top_n (int): The number of top genes to plot. Default is 5.
    """
    non_mt_genes = adata.var[~adata.var.index.str.startswith('mt-')]
    top_genes = non_mt_genes['Raw_Reads'].sort_values(ascending=False).head(top_n).index
    #top_genes = adata.var['Raw_Reads'].sort_values(ascending=False).head(top_n).index
    for gene in top_genes:
        data = adata[:, gene].X.toarray().flatten()
        
        mean = np.mean(data)
        var = np.var(data)
        p = mean / var
        n = mean * p / (1 - p)
        
        theoretical_quantiles = np.linspace(0, 1, len(data))
        theoretical_values = stats.nbinom.ppf(theoretical_quantiles, n, p)
        
        sorted_data = np.sort(data)
        
        plt.figure(figsize=(6, 4))
        plt.plot(theoretical_values, sorted_data, 'o')
        plt.plot(theoretical_values, theoretical_values, 'r--')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Empirical Quantiles')
        plt.title(f'QQ plot for Negative Binomial distribution of {gene}')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(f'{save_path}_{gene}.png')
        plt.show()

def plot_top_n_distr(adata, top_n=3, save_path=None):
    non_mt_genes = adata.var[~adata.var.index.str.startswith('mt-')]
    top_genes = non_mt_genes['Raw_Reads'].sort_values(ascending=False).head(top_n).index
    top_genes_data = adata[:, top_genes].X
    top_genes_df = pd.DataFrame(top_genes_data.toarray(), columns=top_genes, index=adata.obs_names)
    for gene in top_genes:
        plt.figure(figsize=(6, 4))
        sns.histplot(top_genes_df[gene], bins=50, kde=False, label=gene)
        plt.xlabel('Number of reads')
        plt.xlim(0, 500)
        plt.ylabel('Number of cells')
        plt.title(f'{gene} - Number of reads per cell')
        plt.legend(title='Genes')
        if save_path:
            plt.savefig(f'{save_path}_{gene}.png')
        plt.show()

def elbow_plot(adata, save_path=None):
    sc.tl.pca(adata, svd_solver='arpack', n_comps=50, use_highly_variable=True)
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

def plot_top_genes_pca_heatmaps(adata, n_cells=500, n_top_genes=10, pc_index='10m', n_comps=50, random_seed=42, save_path=None):
    """
    Samples random cells and plots heatmaps of PC values for top genes of specified PCs.

    Parameters:
        adata (AnnData): The annotated data matrix.
        n_cells (int): Number of random cells to sample.
        n_top_genes (int): Number of top genes to display for each specified PC.
        pc_index (str or int): Index of the principal component(s) to analyze (0-based), or a string with 'm' to specify multiple PCs.
        n_comps (int): Number of principal components to compute.
        random_seed (int): Seed for random number generator for reproducibility.

    Returns:
        None
    """
    # Sample random cells
    np.random.seed(random_seed)  # For reproducibility
    random_cells = np.random.choice(adata.obs_names, n_cells, replace=False)
    adata_sampled = adata[random_cells, :]

    # Perform PCA on the sampled data
    sc.tl.pca(adata_sampled, svd_solver='arpack', n_comps=n_comps)

    # Retrieve PCA loadings
    pca_loadings = adata_sampled.varm['PCs']
    gene_contributions = pd.DataFrame(pca_loadings, index=adata_sampled.var_names)

    # Determine the PCs to plot
    if isinstance(pc_index, str) and pc_index.endswith('m'):
        num_pcs = int(pc_index[:-1])
        pcs_to_plot = range(num_pcs)
    else:
        pcs_to_plot = [int(pc_index)]

    # Number of rows and columns for subplots
    n_cols = min(len(pcs_to_plot), 5)
    n_rows = (len(pcs_to_plot) + n_cols - 1) // n_cols

    # Create heatmaps for each specified PC
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.5 * n_rows))
    axes = axes.flatten()

    for i, pc in enumerate(pcs_to_plot):
        top_genes = gene_contributions.iloc[:, pc].abs().sort_values(ascending=False).index[:n_top_genes]
        top_gene_values = adata_sampled[:, top_genes].X

        sns.heatmap(top_gene_values.T, cmap='viridis', xticklabels=False, yticklabels=top_genes, ax=axes[i])
        axes[i].set_xlabel('Sampled Cells')
        axes[i].set_ylabel('Top Genes')
        axes[i].set_title(f'Heatmap of PC{pc+1}')

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
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

def plot_tsne(adata, cluster_type='cluster_class_name', legend_fontsize=5, save_path=None):
    sample_tags = adata.obs['Sample_Tag'].unique()
    for tag in sample_tags:
        adata_subset = adata[adata.obs['Sample_Tag'] == tag]
        sc.pl.tsne(
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
        sample_tag_counts.to_excel(save_path)
        
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