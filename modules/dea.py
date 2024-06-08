import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def differential_expression_analysis(sc_df1, sc_df2):
    # Ensure that the indices and columns are strings
    sc_df1.index = sc_df1.index.astype(str)
    sc_df2.index = sc_df2.index.astype(str)
    sc_df1.columns = sc_df1.columns.astype(str)
    sc_df2.columns = sc_df2.columns.astype(str)

    # Combine the data into a single DataFrame and ensure the data is float
    count_data = pd.concat([sc_df1, sc_df2], axis=0).astype(float)

    # Create a sample information DataFrame with condition labels
    sample_info = pd.DataFrame({
        'condition': ['WT'] * sc_df1.shape[0] + ['TG'] * sc_df2.shape[0]
    }, index=count_data.index)

    # Ensure the sample_info index is a string
    sample_info.index = sample_info.index.astype(str)

    # Create an AnnData object with count data and sample information
    adata = sc.AnnData(count_data.values, obs=sample_info)

    # Set the variable names (gene names)
    adata.var_names = count_data.columns.astype(str)

    # Set the observation names (sample names)
    adata.obs_names = count_data.index.astype(str)

    # Store the raw data in the .raw attribute before transformation
    adata.raw = adata.copy()

    # Initialize lists to store the gene names, fold changes, and p-values
    gene_names = []
    log2_fold_changes = []
    p_values = []

    # Perform differential expression analysis gene by gene
    for gene in tqdm(adata.var_names):
        # Extract expression values for the gene
        expr_values = adata[:, gene].X

        # Split expression values by condition
        expr_values_WT = expr_values[adata.obs['condition'] == 'WT']
        expr_values_TG = expr_values[adata.obs['condition'] == 'TG']

        # Perform Wilcoxon rank-sum test
        stat, p_value = ranksums(expr_values_WT, expr_values_TG)

        # Calculate mean expression values
        mean_WT = np.mean(expr_values_WT)
        mean_TG = np.mean(expr_values_TG)

        # Calculate log2 fold change
        if mean_WT > 0 and mean_TG > 0:
            log2_fold_change = np.log2(mean_TG / mean_WT)
        else:
            log2_fold_change = np.nan  # Handle cases where the mean is zero

        # Store the results
        gene_names.append(gene)
        log2_fold_changes.append(log2_fold_change)
        p_values.append(p_value)

    # Convert p-values list to a NumPy array
    p_values = np.array(p_values).reshape(-1)

    # Perform Benjamini-Hochberg correction
    _, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')

    # Calculate log10 of corrected p-values
    log10_pvals_corrected = -np.log10(np.maximum(p_values_corrected, 1e-300))

    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'gene': gene_names,
        'log2foldchange': log2_fold_changes,
        'pval': p_values,
        'pval_corrected': p_values_corrected,
        'log10pval_corrected': log10_pvals_corrected
    })

    return results_df


def get_volcano_plot(results_df, min_fold_change=1, max_p_value=0.05):
    # Copy the DataFrame to avoid modifying the original one
    df = results_df.copy()

    # Identify significant results
    df['significant'] = df['pval_corrected'] < max_p_value

    # Identify results outside the range
    df['outside_range'] = df['significant'] & (df['log2foldchange'].abs() > min_fold_change)

    # Assign colors based on the conditions
    df['color'] = 'grey'
    df.loc[df['outside_range'] & (df['log2foldchange'] > 0), 'color'] = 'red'
    df.loc[df['outside_range'] & (df['log2foldchange'] <= 0), 'color'] = 'blue'

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['log2foldchange'], df['log10pval_corrected'], s=5, c=df['color'])

    # Add lines to the plot
    plt.axvline(x=-min_fold_change, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(x=min_fold_change, color='black', linestyle='--', linewidth=0.5)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    # Annotate the top 20 significant genes
    for _, row in df[df['outside_range']].nlargest(20, 'log10pval_corrected').iterrows():
        plt.annotate(row['gene'], (row['log2foldchange'] + 0.2, row['log10pval_corrected']), ha='left', va='center', fontsize=7)

    # Get the highest and lowest fold changes and the maximum p-value
    high_fold_change = df['log2foldchange'].max()
    low_fold_change = df['log2foldchange'].min()
    max_log10_pval = df['log10pval_corrected'].max()

    # Annotate the number of differentially expressed genes (DEGs) in red and blue
    plt.annotate(f"{df[df['color'] == 'red'].shape[0]} DEGs", xy=(high_fold_change, max_log10_pval), ha='right', va='top', fontsize=10, color='red')
    plt.annotate(f"{df[df['color'] == 'blue'].shape[0]} DEGs", xy=(low_fold_change, max_log10_pval), ha='left', va='top', fontsize=10, color='blue')

    # Add grid, labels and show the plot
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.2)
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-log10(p-value)')
    plt.show()

def get_heatmap(results_df):
    deg = results_df[(results_df["pval_corrected"] < 0.05) & (results_df["log2foldchange"].abs() > 0.25)]['gene'].tolist()

    adata1 = anndata.AnnData(X=sc_df1.loc[:, deg].values, obs=pd.DataFrame(index=sc_df1.index), var=pd.DataFrame(index=deg))
    adata2 = anndata.AnnData(X=sc_df2.loc[:, deg].values, obs=pd.DataFrame(index=sc_df2.index), var=pd.DataFrame(index=deg))
    
    adata1.obs['batch'], adata2.obs['batch'] = 'WT-DMSO', '3xTg-DMSO'
    
    combined_adata = anndata.concat([adata1, adata2])
    sc.pp.scale(combined_adata)
    
    mean_expr = combined_adata.to_df().groupby(combined_adata.obs['batch']).mean().T
    mean_expr['abs_diff_to_min'] = (mean_expr['3xTg-DMSO'] - mean_expr['WT-DMSO']).abs()
    mean_expr_sorted = mean_expr.sort_values(by='abs_diff_to_min', ascending=False).drop(columns='abs_diff_to_min')
    
    sns.set(context='notebook', font_scale=1.2)
    cg = sns.clustermap(mean_expr_sorted, cmap='coolwarm', linewidths=.5, figsize=(10, 15), row_cluster=True, col_cluster=False)
    
    cg.ax_heatmap.set_yticks(range(len(mean_expr_sorted.index)))
    cg.ax_heatmap.set_yticklabels(mean_expr_sorted.index, rotation=0, fontsize=10)
    cg.ax_heatmap.yaxis.set_label_position('right')
    cg.ax_heatmap.yaxis.tick_right()
    
    cg.cax.set_position([1, .2, .03, .45])
    cg.ax_heatmap.xaxis.set_label_position('top')
    cg.ax_heatmap.xaxis.tick_top()
    cg.ax_heatmap.set_xlabel('')
    for tick in cg.ax_heatmap.get_xticklabels():
        tick.set_color('black')
        tick.set_ha('center')
        tick.set_rotation(0)
    
    plt.show()