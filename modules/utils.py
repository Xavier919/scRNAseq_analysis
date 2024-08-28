import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
from collections import Counter
from MCML import tools as tl
from rpy2.robjects import pandas2ri, r
from scipy.sparse import issparse, csr_matrix
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects import default_converter
import anndata2ri

import matplotlib.pyplot as plt
import rpy2.robjects as ro

def analyze_neighbor_fractions(adata, embedding_methods, n_neighbors=30, n_common_types=15):
    """
    Analyze neighbor fractions for different embedding methods.
    
    Parameters:
    adata (AnnData): Annotated data matrix.
    embedding_methods (list): List of embedding method names to analyze.
    n_neighbors (int): Number of neighbors to consider (default 30).
    n_common_types (int): Number of most common cell types to include (default 15).
    
    Returns:
    pandas.DataFrame: Combined results of neighbor fraction analysis.
    """
    subclass_name = adata.obs['subclass_name'].values.tolist()
    
    results = {}
    for method in embedding_methods:
        neighbor_fracs, _ = tl.frac_unique_neighbors(
            adata.obsm[f'X_{method}'], np.array(subclass_name), metric=1, neighbors=n_neighbors
        )
        results[method] = neighbor_fracs
    
    # Identify the most common cell types
    common_cell_types = [x[0] for x in Counter(adata.obs['subclass_name']).most_common(n_common_types)]
    
    # Filter fractions to include only the most common cell types and calculate the mean for each type
    for method in embedding_methods:
        results[method] = {x: np.mean(y) for x, y in results[method].items() if x in common_cell_types}
    
    # Combine the results into a single DataFrame
    combined_df = pd.DataFrame(results)
    
    return combined_df

def annotate_adata(adata, anno_df):
    """
    Annotates the given AnnData object with the provided annotation dataframe.

    Args:
        adata (AnnData): The AnnData object to be annotated.
        anno_df (pd.DataFrame): The annotation dataframe containing the annotations for each cell.

    Returns:
        AnnData: The annotated AnnData object.
    """
    anno_df = anno_df.set_index('cell_id')[['class_name', "subclass_name", "supertype_name", 'cluster_name']]
    adata.obs.index = adata.obs.index.astype(str)
    anno_df.index = anno_df.index.astype(str)
    adata.obs['class_name'] = anno_df['class_name'].apply(lambda x: x.split(' ')[1])
    adata.obs['subclass_name'] = anno_df['subclass_name'].apply(lambda x: x.split(' ')[1])
    adata.obs['supertype_name'] = anno_df['supertype_name'].apply(lambda x: x.split(' ')[1])
    adata.obs['cluster_name'] = anno_df['cluster_name'].apply(lambda x: x.split(' ')[1])
    return adata

def show_pc_variance(adata, layer_name, pc_list=[10,20,50,100]):
    """
    Calculate and print the explained variance for the specified number of principal components (PCs).

    Parameters:
        adata (AnnData): Annotated data object.
        layer_name (str): Name of the layer to perform PCA on.
        pc_list (list, optional): List of integers specifying the number of PCs to calculate the explained variance for. 
            Defaults to [10, 20, 50, 100].

    Returns:
        None
    """
    sc.tl.pca(adata, svd_solver='arpack', n_comps=100, use_highly_variable=True, layer=layer_name)
    pca_variance_ratio = adata.uns['pca']['variance_ratio']
    for pc in pc_list:
        print(f'{layer_name} explained variance for the first {pc} PCs: {np.sum(pca_variance_ratio[:pc])}')

def pearson_normalization(adata):
    """
    Perform Pearson normalization on the input AnnData object.

    Parameters:
        adata (AnnData): The input AnnData object.

    Returns:
        AnnData: The normalized AnnData object.
    """
    analytic_pearson = sc.experimental.pp.normalize_pearson_residuals(adata, inplace=False)
    adata.layers["analytic_pearson_residuals"] = csr_matrix(analytic_pearson["X"])
    
    residuals_sparse = adata.layers["analytic_pearson_residuals"]
    if np.any(np.isnan(residuals_sparse.data)):
        print("Warning: NaNs found in the analytic Pearson residuals.")
        residuals_sparse.data = np.nan_to_num(residuals_sparse.data)
    
    residuals_sum = residuals_sparse.sum(axis=1).A1
    return adata

def scran_normalization(adata):
    """
    Perform scran normalization on the input AnnData object.

    Parameters:
        adata (AnnData): The input AnnData object containing the gene expression data.

    Returns:
        AnnData: The normalized AnnData object.

    Raises:
        None

    Example:
        adata_normalized = scran_normalization(adata)
    """
    adata_pp = adata.copy()
    sc.pp.normalize_total(adata_pp)
    sc.pp.log1p(adata_pp)
    sc.pp.pca(adata_pp, n_comps=15)
    sc.pp.neighbors(adata_pp)
    sc.tl.leiden(adata_pp, key_added="groups")
    
    data_mat = adata_pp.X.T
    
    if issparse(data_mat):
        if data_mat.nnz > 2**31 - 1:
            data_mat = data_mat.tocoo()
        else:
            data_mat = data_mat.tocsc()
    
    data_mat_dense = data_mat.toarray()
    ro.globalenv["data_mat"] = pandas2ri.py2rpy(pd.DataFrame(data_mat_dense))
    ro.globalenv["input_groups"] = pandas2ri.py2rpy(adata_pp.obs["groups"])
    
    ro.r('''
    size_factors <- sizeFactors(
        computeSumFactors(
            SingleCellExperiment(
                list(counts = as.matrix(data_mat))), 
                clusters = input_groups,
                min.mean = 0.1,
                BPPARAM = MulticoreParam()
        )
    )
    ''')
    
    size_factors = ro.globalenv["size_factors"]
    
    adata.obs["size_factors"] = size_factors
    
    scran_normalized = adata.X / adata.obs["size_factors"].values[:, None]
    
    adata.layers["scran_normalization"] = csr_matrix(sc.pp.log1p(scran_normalized))
    return adata

def select_features(adata):
    """
    Selects features using deviance-based feature selection.

    Args:
        adata (AnnData): Annotated data object containing gene expression data.

    Returns:
        AnnData: Annotated data object with selected features.

    Raises:
        None

    Examples:
        >>> adata = select_features(adata)
    """
    adata.X = adata.X.astype(np.float64)
    for layer in adata.layers:
        adata.layers[layer] = adata.layers[layer].astype('float64')
    with localconverter(default_converter + anndata2ri.converter + pandas2ri.converter + numpy2ri.converter):
        ro.globalenv['adata'] = anndata2ri.py2rpy(adata)
    ro.r('sce <- as(adata, "SingleCellExperiment")')
    ro.r('sce <- devianceFeatureSelection(sce, assay="X")')
    binomial_deviance = np.array(ro.r('rowData(sce)$binomial_deviance')).T
    idx = binomial_deviance.argsort()[-4000:]
    mask = np.zeros(adata.var_names.shape, dtype=bool)
    mask[idx] = True
    adata.var["highly_deviant"] = mask
    adata.var["binomial_deviance"] = binomial_deviance
    return adata


def assign_pseudoreplicates(adata):
    """
    Assign pseudoreplicates to the given AnnData object based on the 'Sample_Tag' column.

    Parameters:
        adata (AnnData): The AnnData object containing the data.

    Returns:
        AnnData: The updated AnnData object with pseudoreplicates assigned.
    """
    # Convert 'Sample_Tag' to string if it's not already
    adata.obs['Sample_Tag'] = adata.obs['Sample_Tag'].astype(str)

    # Create a new 'batch' column based on 'Sample_Tag'
    adata.obs['batch'] = adata.obs['Sample_Tag'].astype(str)

    # Get the unique sample tags
    unique_sample_tags = adata.obs['Sample_Tag'].unique()

    # Create new batch labels
    new_batch_labels = []
    for sample_tag in unique_sample_tags:
        new_batch_labels.extend([f"{sample_tag}_1", f"{sample_tag}_2", f"{sample_tag}_3"])

    # Update the categories to include the new batch labels
    adata.obs['batch'] = adata.obs['batch'].astype('category')
    adata.obs['batch'] = adata.obs['batch'].cat.add_categories(new_batch_labels)

    # Loop over each unique sample tag and assign pseudoreplicates
    for sample_tag in unique_sample_tags:
        # Get the indices for the current sample tag
        indices = adata.obs[adata.obs['Sample_Tag'] == sample_tag].index

        # Assign pseudoreplicates randomly
        n = len(indices)
        replicate_labels = np.random.choice([f"{sample_tag}_1", f"{sample_tag}_2", f"{sample_tag}_3"], size=n)

        # Update the batch column with these pseudoreplicate labels
        adata.obs.loc[indices, 'batch'] = replicate_labels

    return adata


def plot_cell_type_abundances(adata, save_path=None):
    """
    Plot the abundances of different cell types based on the provided AnnData object.

    Parameters:
        adata (AnnData): The AnnData object containing the single-cell RNA-seq data.
        save_path (str, optional): The file path to save the plot. If not provided, the plot will be displayed.

    Returns:
        None
    """
    # Extract relevant data
    df = adata.obs[['total_counts', 'Sample_Tag', 'subclass_name']]

    # Aggregate the data to get the number of cells per cell type
    cell_type_counts = df['subclass_name'].value_counts()

    # Filter out cell types with less than 100 cells
    filtered_cell_types = cell_type_counts[cell_type_counts >= 100].index

    # Filter the original DataFrame
    filtered_df = df[df['subclass_name'].isin(filtered_cell_types)]

    # Pivot the DataFrame to get the number of cells instead of total counts
    pivot_df = filtered_df.groupby(['Sample_Tag', 'subclass_name']).size().unstack(fill_value=0)

    # Resetting the index
    pivot_df.reset_index(inplace=True)

    # Melting the DataFrame to long format
    plot_data_global = pivot_df.melt(id_vars="Sample_Tag", var_name="Cell type", value_name="count")

    # Renaming 'Sample_Tag' to 'Condition' to match the plot example
    plot_data_global.rename(columns={'Sample_Tag': 'Condition'}, inplace=True)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Filtered DataFrame for plotting to ensure legends and data match
    filtered_plot_data_global = plot_data_global[
        plot_data_global['Cell type'].isin(filtered_cell_types)
    ]

    # Plot for Global abundances, by condition
    sns.barplot(
        data=filtered_plot_data_global, x="Condition", y="count", hue="Cell type", ax=ax[0]
    )
    ax[0].set_title("Abundances, by condition")
    ax[0].legend(title="Cell type", loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot for Global abundances, by cell type
    sns.barplot(
        data=filtered_plot_data_global, x="Cell type", y="count", hue="Condition", ax=ax[1]
    )
    ax[1].set_title("Abundances, by cell type")
    ax[1].legend(title="Condition", loc='center left', bbox_to_anchor=(1, 0.5))

    # Rotate the x-axis labels of the right plot
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()