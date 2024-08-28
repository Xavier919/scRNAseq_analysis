import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import re
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse as sparse
from scipy import stats

def display_transformation(adata, layer_name, save_path):
    """
    Display the transformation of data using histograms.

    Parameters:
        adata (AnnData): Annotated data object.
        layer_name (str): Name of the layer to visualize.
        save_path (str): Path to save the figure.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    p1 = sns.histplot(adata.obs["total_counts"], bins=100, kde=False, ax=axes[0])
    axes[0].set_title("Total counts")
    p2 = sns.histplot(adata.layers[layer_name].sum(1), bins=100, kde=False, ax=axes[1])
    axes[1].legend_.remove()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def pie_chart_condition(list_, min_pct=0.05, save_path=None):
    """
    Generate a pie chart based on the given list of conditions.

    Args:
        list_ (list): A list of conditions.
        save_path (str, optional): The file path to save the chart image. Defaults to None.

    Returns:
        None
    """
    string_counts = Counter(list_)
    total_count = sum(string_counts.values())
    labels = []
    sizes = []
    other_size = 0
    
    for label, size in string_counts.items():
        percentage = size / total_count
        if percentage >= min_pct:  # min_pct threshold
            labels.append(label)
            sizes.append(size)
        else:
            other_size += size
    
    if other_size > 0:
        labels.append('Others')
        sizes.append(other_size)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = cm.viridis([i / len(labels) for i in range(len(labels))])
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.axis('equal')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
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
    
    fig, axes = plt.subplots(1, top_n, figsize=(6 * top_n, 4))
    
    if top_n == 1:
        axes = [axes]
    
    for i, gene in enumerate(top_genes):
        data = adata[:, gene].X.toarray().flatten()
        
        mean = np.mean(data)
        var = np.var(data)
        p = mean / var
        n = mean * p / (1 - p)
        
        theoretical_quantiles = np.linspace(0, 1, len(data))
        theoretical_values = stats.nbinom.ppf(theoretical_quantiles, n, p)
        
        sorted_data = np.sort(data)
        
        axes[i].plot(theoretical_values, sorted_data, 'o')
        axes[i].plot(theoretical_values, theoretical_values, 'r--')
        axes[i].set_xlabel('Theoretical Quantiles')
        axes[i].set_ylabel('Empirical Quantiles')
        axes[i].set_title(f'QQ plot for Negative Binomial distribution of {gene}')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()


def plot_top_n_distr(adata, top_n=3, save_path=None):
    """
    Plot the distribution of the top n genes in a single-cell RNA-seq dataset.

    Parameters:
        adata (AnnData): Annotated data object containing the single-cell RNA-seq data.
        top_n (int): Number of top genes to plot. Default is 3.
        save_path (str): Path to save the plot. If not provided, the plot will be displayed.

    Returns:
        None
    """
    non_mt_genes = adata.var[~adata.var.index.str.startswith('mt-')]
    top_genes = non_mt_genes['Raw_Reads'].sort_values(ascending=False).head(top_n).index
    top_genes_data = adata[:, top_genes].X
    top_genes_df = pd.DataFrame(top_genes_data.toarray(), columns=top_genes, index=adata.obs_names)
    
    fig, axes = plt.subplots(1, top_n, figsize=(6 * top_n, 4))
    
    if top_n == 1:
        axes = [axes]
    
    for i, gene in enumerate(top_genes):
        sns.histplot(top_genes_df[gene], bins=50, kde=False, label=gene, ax=axes[i])
        axes[i].set_xlabel('Number of reads')
        axes[i].set_xlim(0, 500)
        axes[i].set_ylabel('Number of cells')
        axes[i].set_title(f'{gene} - Number of reads per cell')
        axes[i].legend(title='Genes')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()


def scree_plot(adata, layer='log1p_norm', save_path=None):
    """
    Generate a scree plot for the principal components analysis (PCA) of the given AnnData object.

    Parameters:
        adata (AnnData): The AnnData object containing the data.
        layer (str, optional): The layer of the data to use for PCA. Defaults to 'log1p_norm'.
        save_path (str, optional): The file path to save the plot. Defaults to None.

    Returns:
        None
    """
    sc.tl.pca(adata, svd_solver='arpack', n_comps=50, use_highly_variable=True, layer=layer)
    pca_variance_ratio = adata.uns['pca']['variance_ratio']
    fig, ax = plt.subplots()
    ax.bar(range(len(pca_variance_ratio)), pca_variance_ratio, alpha=0.6)
    ax.plot(range(len(pca_variance_ratio)), pca_variance_ratio, 'o', color='red')
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Variance ratio')
    ax.set_title('Scree plot')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_top_genes_pca_heatmaps(adata, layer='scran_normalization', n_cells=500, n_top_genes=10, pc_index='10m', n_comps=50, random_seed=42, save_path=None):
    """
    Samples random cells and plots heatmaps of PC values for top genes of specified PCs.

    Parameters:
        adata (AnnData): The annotated data matrix.
        n_cells (int): Number of random cells to sample.
        n_top_genes (int): Number of top genes to display for each specified PC.
        pc_index (str or int): Index of the principal component(s) to analyze (0-based), or a string with 'm' to specify multiple PCs.
        n_comps (int): Number of principal components to compute.
        random_seed (int): Seed for random number generator for reproducibility.
        save_path (str or None): Path to save the heatmap plot.

    Returns:
        None
    """
    # Ensure the data has highly variable genes calculated
    if 'highly_variable' not in adata.var:
        raise ValueError("The AnnData object does not contain 'highly_variable' information. Please calculate highly variable genes first.")

    # Subset the data to only highly variable genes
    adata_hvg = adata[:, adata.var['highly_variable']]

    # Sample random cells
    np.random.seed(random_seed)  # For reproducibility
    random_cells = np.random.choice(adata_hvg.obs_names, n_cells, replace=False)
    adata_sampled = adata_hvg[random_cells, :]

    # Perform PCA on the sampled data
    sc.tl.pca(adata_sampled, svd_solver='arpack', n_comps=n_comps, use_highly_variable=True, layer=layer)

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

        if sparse.issparse(top_gene_values):
            top_gene_values = top_gene_values.toarray()

        sns.heatmap(top_gene_values.T, cmap='viridis', xticklabels=False, yticklabels=top_genes, ax=axes[i])
        axes[i].set_xlabel('Sampled Cells')
        axes[i].set_ylabel('Top Genes')
        axes[i].set_title(f'Heatmap of PC{pc+1}')

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_umap(adata, cluster_type='cluster_class_name', legend_fontsize=5, save_path=None):
    """
    Plot UMAP visualization for each sample tag in the given AnnData object.

    Parameters:
        adata (AnnData): The AnnData object containing the data to be visualized.
        cluster_type (str): The column name in `adata.obs` that represents the cluster type.
                            Default is 'cluster_class_name'.
        legend_fontsize (int): The font size of the legend. Default is 5.
        save_path (str): The path to save the generated plots. Default is None.

    Returns:
        None
    """
    sample_tags = adata.obs['Sample_Tag'].unique()
    n_tags = len(sample_tags)
    
    # Determine the grid size for subplots
    n_cols = int(n_tags**0.5)
    n_rows = (n_tags + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()  # Flatten the array of axes for easy iteration
    
    for i, tag in enumerate(sample_tags):
        ax = axes[i]
        adata_subset = adata[adata.obs['Sample_Tag'] == tag]
        sc.pl.umap(
            adata_subset,
            color=[cluster_type],
            size=20,
            title=f'{tag}',
            ax=ax,  # Pass the subplot axis to scanpy's plot function
            show=False,
            legend_loc='on data',
            legend_fontsize=legend_fontsize
        )
    
    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'figures/{save_path}', bbox_inches='tight')
    plt.show()


def plot_tsne(adata, cluster_type='cluster_class_name', legend_fontsize=5, save_path=None):
    """
    Plot t-SNE visualization for each sample tag in the given AnnData object.

    Parameters:
        adata (AnnData): The AnnData object containing the data to be visualized.
        cluster_type (str): The column name in adata.obs to use for coloring the t-SNE plot. Default is 'cluster_class_name'.
        legend_fontsize (int): The font size of the legend. Default is 5.
        save_path (str): The path to save the generated plot. Default is None.

    Returns:
        None
    """
    sample_tags = adata.obs['Sample_Tag'].unique()
    n_tags = len(sample_tags)
    
    # Determine the grid size for subplots
    n_cols = int(n_tags**0.5)
    n_rows = (n_tags + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()  # Flatten the array of axes for easy iteration
    
    for i, tag in enumerate(sample_tags):
        ax = axes[i]
        adata_subset = adata[adata.obs['Sample_Tag'] == tag]
        sc.pl.tsne(
            adata_subset,
            color=[cluster_type],
            size=20,
            title=f'{tag}',
            ax=ax,  # Pass the subplot axis to scanpy's plot function
            show=False,
            legend_loc='on data',
            legend_fontsize=legend_fontsize
        )
    
    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
        
    if save_path:
        plt.savefig(f'figures/{save_path}', bbox_inches='tight')
    plt.show()

def remove_numbers(cell_type):
    """
    Removes leading numbers and whitespace from a given cell type.

    Args:
        cell_type (str): The cell type string to process.

    Returns:
        str: The cell type string with leading numbers and whitespace removed.
    """
    return re.sub(r'^\d+\s+', '', cell_type)

def assign_unique_cell_type_names(adata, cluster_key='leiden', cluster_types=['class_name', 'subclass_name']):
    """
    Assigns unique cell type names to each cluster in the AnnData object based on the most common cell type annotation.

    Parameters:
        adata (AnnData): The AnnData object containing the single-cell data.
        cluster_key (str): The key in `adata.obs` that represents the cluster labels.
        cluster_types (list): The list of keys in `adata.obs` that represent the cell type annotations.

    Returns:
        None
    """
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
    """
    Generate a master table containing sample tag counts for each cluster type.

    Parameters:
        adata (AnnData): Annotated data object.
        cluster_type (str, optional): Column name in adata.obs to use as cluster type. Defaults to 'cluster_class_name'.
        save_path (str, optional): File path to save the resulting table as an Excel file. Defaults to None.

    Returns:
        pandas.DataFrame: A DataFrame containing sample tag counts for each cluster type.
    """
    merged_df = adata.obs[['Sample_Tag', cluster_type]]
    result_df = merged_df.groupby(cluster_type).size().reset_index(name='total_count')
    sample_tag_counts = merged_df.groupby([cluster_type, 'Sample_Tag']).size().unstack(fill_value=0)
    
    if save_path is not None:
        sample_tag_counts.to_excel(save_path)
        
    return sample_tag_counts

def create_ditto_plot(adata, sample_tags, class_level, cluster_type, min_cell=50, save_path=None):
    """
    Create a Ditto plot to visualize the cell type composition of clusters.

    Parameters:
    - adata (AnnData): Annotated data object containing the single-cell data.
    - sample_tags (list): List of sample tags to include in the plot.
    - class_level (str): Column name in the adata.obs DataFrame representing the cell type class.
    - cluster_type (str): Column name in the adata.obs DataFrame representing the cluster.
    - min_cell (int, optional): Minimum number of cells required for a class to be included in the plot. Default is 50.
    - save_path (str, optional): File path to save the plot. If not provided, the plot will be displayed.

    Returns:
    None
    """
    df = adata.obs[['Sample_Tag', class_level, cluster_type]]
    class_counts = df[class_level].value_counts()
    valid_classes = class_counts[class_counts >= min_cell].index
    colormap = plt.get_cmap('tab20')  
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