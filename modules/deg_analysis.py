import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
import re
from goatools.associations import read_ncbi_gene2go
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
from goatools.obo_parser import GODag
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.test_data.genes_NCBI_10090_ProteinCoding import GENEID2NT as MOUSE_GENEID2NT
import mygene

def query_genes(adata):
    reference_list = adata.var.index.str.upper().tolist()
    mg = mygene.MyGeneInfo()
    background_gene_info = mg.querymany(reference_list, scopes='symbol', fields='entrezgene', species='mouse')
    gene_to_ncbi = {entry['query']: entry.get('entrezgene') for entry in background_gene_info if 'entrezgene' in entry}
    return gene_to_ncbi

def perform_go_enrichment(gene_list, background_genes):
    """
    Perform GO enrichment analysis.

    Parameters:
    gene_list (list): List of gene IDs to analyze.
    background_genes (list): List of background gene IDs.

    Returns:
    pandas.DataFrame: A DataFrame containing the significant GO enrichment results.
    """
    obodag = GODag("DEG/go-basic.obo")
    objanno = Gene2GoReader("DEG/gene2go", taxids=[10090])
    ns2assoc = objanno.get_ns2assc()
    
    goeaobj = GOEnrichmentStudyNS(
        background_genes, 
        ns2assoc, 
        obodag, 
        propagate_counts=False,
        alpha=0.05, 
        methods=['fdr_bh']
    )
    
    goea_results_all = goeaobj.run_study(gene_list)
    goea_results_sig = [r for r in goea_results_all if r.p_fdr_bh < 0.05]
    
    df_results = pd.DataFrame([{
        'GO_ID': r.GO,
        'GO_term': r.name,
        'namespace': r.NS,
        'study_count': r.study_count,
        'population_count': r.pop_count,
        'study_items': list(r.study_items),
        'population_items': list(r.pop_items),
        'p_uncorrected': r.p_uncorrected,
        'p_fdr_bh': r.p_fdr_bh,
        'fold_enrichment': (r.study_count / len(gene_list)) / (r.pop_count / len(background_genes)),
        'enrichment': 'enriched' if r.enrichment else 'purified'
    } for r in goea_results_sig])
    
    return df_results

def DEG_analysis(adata, ctr, cnd, cell_type):
    """
    Performs differential expression analysis for specified cell types between control and condition groups.
    
    Parameters:
        adata (anndata.AnnData): The input AnnData object containing the observations.
        ctr (str): The control sample tag.
        cnd (str): The condition sample tag.
        cell_type (list of str): The list of cell types to be included in the analysis.
    
    Returns:
        de_results_df (pd.DataFrame): DataFrame containing the differential expression results or None if there are insufficient samples.
    """
    # Subset the AnnData object for the specified cell types
    subset_adata = adata[adata.obs['annotated_cluster'].isin(cell_type)].copy()
    
    # Further subset the data for the specified control and condition groups
    subset_adata = subset_adata[subset_adata.obs['Sample_Tag'].isin([cnd, ctr])].copy()
    
    # Check if each group has at least two samples
    group_counts = subset_adata.obs['Sample_Tag'].value_counts()
    if group_counts.get(ctr, 0) < 2 or group_counts.get(cnd, 0) < 2:
        print(f"Insufficient samples for DE analysis in cell type {cell_type}: {group_counts.to_dict()}")
        return None
    
    # Extract raw counts and create a new AnnData object
    subset_adata_raw = subset_adata.raw.to_adata().copy()
    
    # Normalize the data
    sc.pp.normalize_total(subset_adata_raw)
    sc.pp.log1p(subset_adata_raw)
    # Ensure 'Sample_Tag' is treated as a categorical variable
    subset_adata_raw.obs['Sample_Tag'] = subset_adata_raw.obs['Sample_Tag'].astype('category')
    
    # Perform differential expression analysis
    sc.tl.rank_genes_groups(subset_adata_raw, groupby='Sample_Tag', reference=ctr, method='wilcoxon', corr_method='benjamini-hochberg')
    
    # Extract results for the condition group
    de_results_df = sc.get.rank_genes_groups_df(subset_adata_raw, group=cnd)
    
    return de_results_df

def horizontal_deg_chart(data, fig_tag):
    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=['Category', 'Positive', 'Negative'])
    # Make negative values for Negative DEGs
    df['Negative'] = -df['Negative']
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot the bars
    ax.barh(df['Category'], df['Positive'], color='red', label='Positive DEGs')
    ax.barh(df['Category'], df['Negative'], color='blue', label='Negative DEGs')
    # Add labels
    ax.set_xlabel('Number of DEGs')
    ax.set_ylabel('Clusters')
    ax.legend()
    # Draw vertical line at zero
    ax.axvline(x=0, color='black', linewidth=0.8)
    # Set x-axis limits to center at zero and show 250 on both sides
    #ax.set_xlim(-250, 250)
    # Remove negative sign from x-axis labels
    x_labels = ax.get_xticks()
    ax.set_xticklabels([abs(int(x)) for x in x_labels])
    plt.savefig(f'figures/DEG_horizontal_chart_{fig_tag}.png')
    plt.show()

def get_volcano_plot(results_df, fig_tag, min_fold_change=1, max_p_value=0.01):
    df = results_df.copy()

    # Exclude genes with fold change outside of -10, 10
    df = df[(df['logfoldchanges'] >= -10) & (df['logfoldchanges'] <= 10)]

    # Remove genes with adjusted p-value of 0
    df = df[df['pvals_adj'] != 0]

    # Calculate -log10 of adjusted p-values
    df['log10pval_corrected'] = -np.log10(df['pvals_adj'])

    # Identify significant genes
    df['significant'] = df['pvals_adj'] < max_p_value

    # Identify genes outside the specified fold change range
    df['outside_range'] = df['significant'] & (df['logfoldchanges'].abs() > min_fold_change)

    # Assign colors for plotting
    df['color'] = 'grey'
    df.loc[df['outside_range'] & (df['logfoldchanges'] > 0), 'color'] = 'red'
    df.loc[df['outside_range'] & (df['logfoldchanges'] <= 0), 'color'] = 'blue'

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['logfoldchanges'], df['log10pval_corrected'], s=5, c=df['color'])

    # Add threshold lines
    plt.axvline(x=-min_fold_change, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(x=min_fold_change, color='black', linestyle='--', linewidth=0.5)
    plt.axhline(y=-np.log10(max_p_value), color='black', linestyle='--', linewidth=0.5)

    # Annotate top 20 significant genes
    for _, row in df[df['outside_range']].nlargest(20, 'log10pval_corrected').iterrows():
        plt.annotate(row['names'], (row['logfoldchanges'] + 0.2, row['log10pval_corrected']), ha='left', va='center', fontsize=5)

    # Annotate the number of DEGs
    high_fold_change = df['logfoldchanges'].max()
    low_fold_change = df['logfoldchanges'].min()
    max_log10_pval = df['log10pval_corrected'].max()

    plt.annotate(f"{df[df['color'] == 'red'].shape[0]} DEGs", xy=(high_fold_change, max_log10_pval), ha='right', va='top', fontsize=10, color='red')
    plt.annotate(f"{df[df['color'] == 'blue'].shape[0]} DEGs", xy=(low_fold_change, max_log10_pval), ha='left', va='top', fontsize=10, color='blue')

    # Add grid, labels, and title
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.2)
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-log10(p-value)')
    plt.title('Volcano plot')
    plt.savefig(f'figures/volcano_plot_{fig_tag}.png')
    plt.show()

def get_heatmap(adata, sampletag1, sampletag2, fig_tag, results_df, display_top_n=100, min_fold_change=0.25, max_p_value=0.05):
    # Filter DEGs
    deg = results_df[(results_df["pvals_adj"] < max_p_value) & (results_df["logfoldchanges"].abs() > min_fold_change)]['names'].tolist()
    
    # Create a raw AnnData object
    adata_raw = anndata.AnnData(X=adata.raw.X, obs=adata.obs, var=adata.raw.var)
    
    # Normalize the raw data
    sc.pp.normalize_total(adata_raw)
    
    # Extract the data frame for mean expression
    mean_expr = adata_raw.to_df().groupby(adata_raw.obs['Sample_Tag']).mean().T
    
    # Subset the DataFrame to include only the desired sample tags
    mean_expr = mean_expr[[sampletag1, sampletag2]]
    
    # Calculate the difference between the two sample tags
    mean_expr['diff'] = mean_expr[sampletag2] - mean_expr[sampletag1]
    
    # Sort the DataFrame by the difference
    mean_expr_sorted = mean_expr.sort_values(by='diff', ascending=False)
    
    # Select top 100 positive and negative DEGs
    top_pos = mean_expr_sorted[mean_expr_sorted['diff'] > 0][:display_top_n]
    top_neg = mean_expr_sorted[mean_expr_sorted['diff'] < 0][:display_top_n]
    mean_expr_sorted = pd.concat([top_pos, top_neg])
    
    # Drop the 'diff' column for clustering
    heatmap_data = mean_expr_sorted.drop(columns='diff')
        
    # Generate the heatmap
    sns.set(context='notebook', font_scale=1.2)
    cg = sns.clustermap(heatmap_data, cmap='coolwarm', linewidths=.5, figsize=(10, 15), row_cluster=True, col_cluster=False)
    
    cg.ax_heatmap.set_yticks(range(len(mean_expr_sorted.index)))
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
    
    # Save and display the heatmap
    plt.savefig(f'figures/deg_heatmap_{fig_tag}.png')
    plt.show()


def display_go_enrichment(df, fig_tag, namespace='BP'):
    df = df[df['namespace'] == namespace]
    df = df[['GO_term', 'fold_enrichment', 'p_fdr_bh']]
    # Calculate -log10(FDR)
    df['-log10(FDR)'] = -np.log10(df['p_fdr_bh'])
    # Select top processes based on -log10(FDR)
    top_processes = df.nlargest(20, '-log10(FDR)')
    # Simplify GO terms
    top_processes['GO_term'] = top_processes['GO_term'].apply(lambda x: re.sub(r'\s*\([^)]*\)', '', x))
    top_processes = top_processes.sort_values(by='-log10(FDR)', ascending=False)
    # Plot
    plt.figure(figsize=(8, 6))
    norm = plt.Normalize(top_processes['-log10(FDR)'].min(), top_processes['-log10(FDR)'].max())
    colors = plt.cm.viridis(norm(top_processes['-log10(FDR)']))
    bars = plt.barh(top_processes['GO_term'], top_processes['fold_enrichment'], color=colors)
    
    plt.xlabel('Fold Enrichment')
    plt.ylabel('Biological Process')
    plt.gca().invert_yaxis()  # To display the highest -log10(FDR) at the top
    
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.ax.set_title('-log10(FDR)', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'figures/go_enrichment_results_{namespace}_{fig_tag}.png')
    plt.show()