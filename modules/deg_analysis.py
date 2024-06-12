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

def DEG_analysis(adata, ctr, cnd, cell_type):
    subset_adata = adata[adata.obs['annotated_cluster'].isin(cell_type)].copy()
    subset_adata = subset_adata[subset_adata.obs['Sample_Tag'].isin([cnd, ctr])].copy()
    subset_adata_raw = subset_adata.raw.to_adata().copy()
    sc.pp.normalize_total(subset_adata_raw, target_sum=1, exclude_highly_expressed=True, max_fraction=0.05)
    subset_adata_raw.obs['Sample_Tag'] = subset_adata_raw.obs['Sample_Tag'].astype('category')
    sc.tl.rank_genes_groups(subset_adata_raw, groupby='Sample_Tag', reference=ctr, method='wilcoxon', corr_method='benjamini-hochberg')
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
    ax.set_xlim(-250, 250)
    # Remove negative sign from x-axis labels
    x_labels = ax.get_xticks()
    ax.set_xticklabels([abs(int(x)) for x in x_labels])
    plt.savefig(f'figures/DEG_horizontal_chart_{fig_tag}.png')
    plt.show()

def get_volcano_plot(results_df, fig_tag, min_fold_change=0.26, max_p_value=0.05):
    df = results_df.copy()

    df['log10pval_corrected'] = -np.log10(df['pvals_adj'].replace(0, np.nextafter(0, 1)))  # Replace 0s to avoid -inf

    df['significant'] = df['pvals_adj'] < max_p_value

    df['outside_range'] = df['significant'] & (df['logfoldchanges'].abs() > min_fold_change)

    df['color'] = 'grey'
    df.loc[df['outside_range'] & (df['logfoldchanges'] > 0), 'color'] = 'red'
    df.loc[df['outside_range'] & (df['logfoldchanges'] <= 0), 'color'] = 'blue'

    plt.figure(figsize=(10, 6))
    plt.scatter(df['logfoldchanges'], df['log10pval_corrected'], s=5, c=df['color'])

    plt.axvline(x=-min_fold_change, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(x=min_fold_change, color='black', linestyle='--', linewidth=0.5)
    plt.axhline(y=-np.log10(max_p_value), color='black', linestyle='--', linewidth=0.5)

    for _, row in df[df['outside_range']].nlargest(20, 'log10pval_corrected').iterrows():
        plt.annotate(row['names'], (row['logfoldchanges'] + 0.2, row['log10pval_corrected']), ha='left', va='center', fontsize=7)

    high_fold_change = df['logfoldchanges'].max()
    low_fold_change = df['logfoldchanges'].min()
    max_log10_pval = df['log10pval_corrected'].max()

    plt.annotate(f"{df[df['color'] == 'red'].shape[0]} DEGs", xy=(high_fold_change, max_log10_pval), ha='right', va='top', fontsize=10, color='red')
    plt.annotate(f"{df[df['color'] == 'blue'].shape[0]} DEGs", xy=(low_fold_change, max_log10_pval), ha='left', va='top', fontsize=10, color='blue')

    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.2)
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-log10(p-value)')
    plt.title('Volcano plot')
    plt.savefig(f'figures/volcano_plot_{fig_tag}.png')
    plt.show()

def get_heatmap(results_df, sc_df1, sc_df2, display_top_n=100, fig_tag):
    deg = results_df[(results_df["pval_corrected"] < 0.05) & (results_df["log2foldchange"].abs() > 0.25)]['gene'].tolist()

    adata1 = anndata.AnnData(X=sc_df1.loc[:, deg].values, obs=pd.DataFrame(index=sc_df1.index), var=pd.DataFrame(index=deg))
    adata2 = anndata.AnnData(X=sc_df2.loc[:, deg].values, obs=pd.DataFrame(index=sc_df2.index), var=pd.DataFrame(index=deg))
    
    adata1.obs['batch'], adata2.obs['batch'] = 'WT-DMSO', '3xTg-DMSO'
    
    combined_adata = anndata.concat([adata1, adata2])
    sc.pp.scale(combined_adata)
    
    mean_expr = combined_adata.to_df().groupby(combined_adata.obs['batch']).mean().T
    mean_expr['diff'] = mean_expr['3xTg-DMSO'] - mean_expr['WT-DMSO']
    mean_expr_sorted = mean_expr.sort_values(by='diff', ascending=False)
    
    # Select top 100 positive and negative DEGs
    top_pos = mean_expr_sorted[mean_expr_sorted['diff'] > 0][:display_top_n]
    top_neg = mean_expr_sorted[mean_expr_sorted['diff'] < 0][:display_top_n]
    mean_expr_sorted = pd.concat([top_pos, top_neg])
    
    sns.set(context='notebook', font_scale=1.2)
    cg = sns.clustermap(mean_expr_sorted.drop(columns='diff'), cmap='coolwarm', linewidths=.5, figsize=(10, 15), row_cluster=True, col_cluster=False)
    
    cg.ax_heatmap.set_yticks(range(len(mean_expr_sorted.index)))
    cg.ax_heatmap.set_yticklabels(mean_expr_sorted.index, rotation=0, fontsize=4)
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
    plt.savefig(f'figures/deg_heatmap_{fig_tag}.png')
    plt.show()


def display_go_enrichment(tsv_file_path):
    df = pd.read_csv(tsv_file_path, delimiter='\t', skiprows=11)
    df['upload_1 (fold Enrichment)'] = pd.to_numeric(df['upload_1 (fold Enrichment)'], errors='coerce')
    df['upload_1 (FDR)'] = pd.to_numeric(df['upload_1 (FDR)'], errors='coerce')

    df = df.dropna(subset=['upload_1 (fold Enrichment)', 'upload_1 (FDR)'])
    df = df[df['upload_1 (fold Enrichment)'] >= 5]

    df['-log10(FDR)'] = -np.log10(df['upload_1 (FDR)'])
    top_processes = df.nlargest(20, '-log10(FDR)')

    top_processes['GO biological process complete'] = top_processes['GO biological process complete'].apply(lambda x: re.sub(r'\s*\([^)]*\)', '', x))
    top_processes = top_processes.sort_values(by='-log10(FDR)', ascending=False)

    plt.figure(figsize=(10, 8))
    norm = plt.Normalize(top_processes['-log10(FDR)'].min(), top_processes['-log10(FDR)'].max())
    colors = plt.cm.viridis(norm(top_processes['-log10(FDR)']))
    bars = plt.barh(top_processes['GO biological process complete'], top_processes['upload_1 (fold Enrichment)'], color=colors)

    plt.xlabel('Fold Enrichment')
    plt.ylabel('Biological Process')
    plt.gca().invert_yaxis()  # To display the highest -log10(FDR) at the top

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.ax.set_title('-log10(FDR)', pad=20)

    plt.tight_layout()
    plt.show()


def dump_deg(results_df):
    """
    This function takes a DataFrame of differential expression results, 
    extracts the positively and negatively differentially expressed genes 
    (DEGs) based on certain criteria, and writes these gene names to two 
    separate files.
    """
    positive_deg = [x.upper() + '\n' for x in results_df[(results_df["pval_corrected"] < 0.05) & (results_df["log2foldchange"] > 0.25)]['gene'].tolist()]
    negative_deg = [x.upper() + '\n' for x in results_df[(results_df["pval_corrected"] < 0.05) & (results_df["log2foldchange"] <= -0.25)]['gene'].tolist()]

    with open('DEG/DEG_positive_list', 'w') as file:
        file.writelines(positive_deg)

    with open('DEG/DEG_negative_list', 'w') as file:
        file.writelines(negative_deg)