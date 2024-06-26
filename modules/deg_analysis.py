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
import pickle
import gseapy as gp
from rpy2.robjects import pandas2ri, r
import rpy2.robjects as robjects
from sklearn.utils import resample


def kegg_enrichment_analysis(gene_list, save_path=None):
    enr = gp.enrichr(gene_list=gene_list,
                     gene_sets='KEGG_2019_Mouse',
                     outdir=None, 
                     cutoff=0.05)  

    results = enr.results

    results_sig = results[results['Adjusted P-value'] < 0.05]

    df_results = pd.DataFrame({
        'KEGG_term': results_sig['Term'],
        'study_count': results_sig['Overlap'].apply(lambda x: int(x.split('/')[0])),
        'population_count': results_sig['Overlap'].apply(lambda x: int(x.split('/')[1])),
        'study_items': results_sig['Genes'].str.split(';'),
        'p_uncorrected': results_sig['P-value'],
        'p_fdr_bh': results_sig['Adjusted P-value'],
        'fold_enrichment': results_sig['Combined Score'], 
        'enrichment': results_sig['Combined Score'].apply(lambda x: 'enriched' if x > 1 else 'purified')
    })

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(df_results, f)

    return df_results

def annotate_adata(adata, anno_df):
    anno_df = anno_df.set_index('cell_id')[['class_name', "subclass_name", "supertype_name", 'cluster_name']]
    adata.obs.index = adata.obs.index.astype(str)
    anno_df.index = anno_df.index.astype(str)
    adata.obs['class_name'] = anno_df['class_name']
    adata.obs['subclass_name'] = anno_df['subclass_name']
    adata.obs['supertype_name'] = anno_df['supertype_name']
    adata.obs['cluster_name'] = anno_df['cluster_name']
    return adata

def get_DEGs(df, genes_ncbi, max_pval=0.05, min_fold_change=0.25):
    filtered_above = df[(df['padj'] < max_pval) & (df['log2FoldChange'] > min_fold_change)]
    filtered_below = df[(df['padj'] < max_pval) & (df['log2FoldChange'] < -min_fold_change)]
    genes_above_id = [genes_ncbi[x.upper()] for x in filtered_above['names'] if x.upper() in genes_ncbi]
    genes_below_id = [genes_ncbi[x.upper()] for x in filtered_below['names'] if x.upper() in genes_ncbi]
    genes_above_name = [x.upper() for x in filtered_above['names'] if x.upper()]
    genes_below_name = [x.upper() for x in filtered_below['names'] if x.upper()]
    return genes_above_id, genes_below_id, genes_above_name, genes_below_name

def query_genes(adata, save_path=None):
    reference_list = adata.var.index.str.upper().tolist()
    mg = mygene.MyGeneInfo()
    background_gene_info = mg.querymany(reference_list, scopes='symbol', fields='entrezgene', species='mouse')
    gene_to_ncbi = {entry['query']: int(entry.get('entrezgene')) for entry in background_gene_info if 'entrezgene' in entry}
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(gene_to_ncbi, f)
    return gene_to_ncbi

def go_enrichment_analysis(gene_list, background_genes, save_path=None):
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
    
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(save_path, f)
    return df_results

def go_enrichment_analysis(gene_list, save_path=None):
    categories = {
        'BP': 'GO_Biological_Process_2021',
        'MF': 'GO_Molecular_Function_2021',
        'CC': 'GO_Cellular_Component_2021'
    }
    
    all_results = []

    for namespace, gene_set in categories.items():
        enr = gp.enrichr(gene_list=gene_list,
                         gene_sets=gene_set,  
                         outdir=None, 
                         cutoff=0.05)  

        results = enr.results

        # Filter results by adjusted p-value
        results_sig = results[results['Adjusted P-value'] < 0.05]

        # Create a DataFrame with the relevant information
        df_results = pd.DataFrame({
            'GO_term': results_sig['Term'],
            'study_count': results_sig['Overlap'].apply(lambda x: int(x.split('/')[0])),
            'population_count': results_sig['Overlap'].apply(lambda x: int(x.split('/')[1])),
            'study_items': results_sig['Genes'].str.split(';'),
            'p_uncorrected': results_sig['P-value'],
            'p_fdr_bh': results_sig['Adjusted P-value'],
            'fold_enrichment': results_sig['Combined Score'], 
            'enrichment': results_sig['Combined Score'].apply(lambda x: 'enriched' if x > 1 else 'purified'),
            'namespace': namespace
        })
        
        all_results.append(df_results)
    
    # Concatenate all results into a single DataFrame
    final_results = pd.concat(all_results, ignore_index=True)

    # Save results to a file if a save path is provided
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(final_results, f)

    return final_results

def DEG_analysis(adata, ctr, cnd, cell_type, save_path=None):
    subset_adata = adata[adata.obs['cluster_class_name'].isin(cell_type)].copy()
    subset_adata = subset_adata[subset_adata.obs['Sample_Tag'].isin([cnd, ctr])].copy()
    group_counts = subset_adata.obs['Sample_Tag'].value_counts()
    if group_counts.get(ctr, 0) < 2 or group_counts.get(cnd, 0) < 2:
        print(f"Insufficient samples for DE analysis in cell type {cell_type}: {group_counts.to_dict()}")
        return None
    subset_adata_raw = subset_adata.raw.to_adata().copy()
    sc.pp.normalize_total(subset_adata_raw)
    sc.pp.log1p(subset_adata_raw)
    subset_adata_raw.obs['Sample_Tag'] = subset_adata_raw.obs['Sample_Tag'].astype('category')
    sc.tl.rank_genes_groups(subset_adata_raw, groupby='Sample_Tag', reference=ctr, method='wilcoxon', corr_method='benjamini-hochberg')
    de_results_df = sc.get.rank_genes_groups_df(subset_adata_raw, group=cnd)
    de_results_df.rename(columns={'logfoldchanges': 'log2FoldChange', 'pvals_adj': 'padj'}, inplace=True)

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(save_path, f)
    return de_results_df

def DEG_analysis_deseq2(adata, ctr, cnd, cell_type, save_path=None):
    pandas2ri.activate()
    ro.r('if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")')
    ro.r('BiocManager::install("DESeq2", update=FALSE)')
    ro.r('library(DESeq2)')
    subset_adata = adata[adata.obs['cluster_class_name'].isin(cell_type)].copy()
    subset_adata = subset_adata[subset_adata.obs['Sample_Tag'].isin([cnd, ctr])].copy()
    group_counts = subset_adata.obs['Sample_Tag'].value_counts()
    if group_counts.get(ctr, 0) < 2 or group_counts.get(cnd, 0) < 2:
        print(f"Insufficient samples for DE analysis in cell type {cell_type}: {group_counts.to_dict()}")
        return None

    counts = subset_adata.raw.X.toarray()
    counts = pd.DataFrame(counts, index=subset_adata.obs_names, columns=subset_adata.raw.var_names)
    metadata = subset_adata.obs[['Sample_Tag']]
    
    r_counts = pandas2ri.py2rpy(counts.T) 
    r_metadata = pandas2ri.py2rpy(metadata)
    
    deseq2_script = """
    library(DESeq2)
    
    run_deseq2 <- function(counts, metadata, control, condition) {
        dds <- DESeqDataSetFromMatrix(countData = counts,
                                      colData = metadata,
                                      design = ~ Sample_Tag)
        dds <- dds[ rowSums(counts(dds)) > 1, ]
        dds$Sample_Tag <- relevel(dds$Sample_Tag, ref = control)
        
        # Estimate size factors using an alternative method
        geoMeans <- apply(counts(dds), 1, function(x) exp(mean(log(x[x > 0]), na.rm = TRUE)))
        dds <- estimateSizeFactors(dds, geoMeans=geoMeans)
        
        dds <- DESeq(dds)
        res <- results(dds, contrast=c("Sample_Tag", condition, control))
        res_df <- as.data.frame(res)
        return(res_df)
    }
    """
    ro.r(deseq2_script)
    
    run_deseq2 = ro.globalenv['run_deseq2']
    res_r = run_deseq2(r_counts, r_metadata, ctr, cnd)
    
    de_results_df = pandas2ri.rpy2py(res_r)

    de_results_df = de_results_df.reset_index().rename(columns={'index': 'names'})
    
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(save_path, f)
    return de_results_df

def DEG_analysis_deseq2(adata, ctr, cnd, cell_type, save_path=None, n_subsamples=5, random_state=42):
    pandas2ri.activate()
    
    # Ensure DESeq2 is installed and loaded
    r('if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")')
    r('BiocManager::install("DESeq2", update=FALSE)')
    r('library(DESeq2)')
    
    # Set the random seed
    np.random.seed(random_state)
    
    # Subset the data for the specified cell type
    subset_adata = adata[adata.obs['cluster_class_name'].isin(cell_type)]
    subset_adata = subset_adata[subset_adata.obs['Sample_Tag'].isin([cnd, ctr])]
    
    # Check sample counts per condition
    group_counts = subset_adata.obs['Sample_Tag'].value_counts()
    if group_counts.get(ctr, 0) < 2 or group_counts.get(cnd, 0) < 2:
        print(f"Insufficient samples for DE analysis in cell type {cell_type}: {group_counts.to_dict()}")
        return None

    # Aggregate counts per condition with subsampling (pseudobulk with replication)
    condition_labels = subset_adata.obs['Sample_Tag'].unique()
    pseudobulk_counts = []
    pseudobulk_labels = []
    
    for condition in condition_labels:
        cells_in_condition = subset_adata[subset_adata.obs['Sample_Tag'] == condition]
        cells_count = cells_in_condition.raw.X
        
        # Adjust n_subsamples if necessary
        actual_subsamples = min(n_subsamples, cells_count.shape[0])
        
        # Create pseudobulk samples
        for i in range(actual_subsamples):
            subsample_indices = resample(np.arange(cells_count.shape[0]), n_samples=max(1, cells_count.shape[0] // actual_subsamples), replace=False, random_state=random_state)
            subsample_counts = cells_count[subsample_indices].sum(axis=0).flatten()
            pseudobulk_counts.append(subsample_counts)
            pseudobulk_labels.append(f"{condition}_subsample_{i+1}")

    # Convert to a 2D numpy array
    pseudobulk_counts = np.vstack(pseudobulk_counts)

    # Create a DataFrame for counts
    counts_df = pd.DataFrame(pseudobulk_counts.T, 
                             columns=pseudobulk_labels, 
                             index=subset_adata.raw.var_names)
    
    # Metadata for conditions
    metadata = pd.DataFrame({'Sample_Tag': pseudobulk_labels, 'Condition': [label.split('_subsample_')[0] for label in pseudobulk_labels]})
    
    # Ensure row and column names are correctly set
    counts_df.index.name = None
    counts_df.columns.name = None
    metadata.index = counts_df.columns
    
    # Convert to R dataframes
    r_counts = pandas2ri.py2rpy(counts_df)
    r_metadata = pandas2ri.py2rpy(metadata)
    
    # DESeq2 analysis script with gene-wise dispersion estimates
    deseq2_script = """
    library(DESeq2)
    
    run_deseq2 <- function(counts, metadata) {
        dds <- DESeqDataSetFromMatrix(countData = counts,
                                      colData = metadata,
                                      design = ~ Condition)
        dds <- dds[ rowSums(counts(dds)) > 1, ]
        dds <- estimateSizeFactors(dds)
        dds <- estimateDispersionsGeneEst(dds)
        dispersions(dds) <- mcols(dds)$dispGeneEst
        dds <- nbinomWaldTest(dds)
        res <- results(dds, contrast=c("Condition", unique(metadata$Condition)[1], unique(metadata$Condition)[2]))
        res_df <- as.data.frame(res)
        return(res_df)
    }
    """
    robjects.r(deseq2_script)
    
    try:
        # Run DESeq2 analysis
        run_deseq2 = robjects.globalenv['run_deseq2']
        res_r = run_deseq2(r_counts, r_metadata)
        
        # Convert results back to a Pandas dataframe
        de_results_df = pandas2ri.rpy2py(res_r)
        de_results_df = de_results_df.reset_index().rename(columns={'index': 'names'})
        
        # Save results if save_path is provided
        if save_path is not None:
            de_results_df.to_csv(save_path, index=False)
        
        return de_results_df
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def horizontal_deg_chart(adata, cell_types, ctr, cnd, min_fold_change=0.25, max_p_value=0.05, fig_title=None, save_path=None):
    cluster_n_DEGs = []

    for cell_type in tqdm(cell_types):
        df = DEG_analysis_deseq2(adata, ctr, cnd, [cell_type])
        if df is None:
            continue
        positive_enriched = df[(df['log2FoldChange'] > min_fold_change) & (df['padj'] < max_p_value)]
        negative_enriched = df[(df['log2FoldChange'] < -min_fold_change) & (df['padj'] < max_p_value)]
        positive_count = positive_enriched.shape[0]
        negative_count = negative_enriched.shape[0]
        cluster_n_DEGs.append((cell_type, positive_count, negative_count))
    
    df = pd.DataFrame(cluster_n_DEGs, columns=['Category', 'Positive', 'Negative'])
    
    df['Negative'] = -df['Negative']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(df['Category'], df['Positive'], color='red', label='Positive DEGs')
    ax.barh(df['Category'], df['Negative'], color='blue', label='Negative DEGs')
    if fig_title is not None:
        ax.set_title(fig_title)
    ax.set_xlabel('Number of DEGs')
    ax.set_ylabel('Clusters')
    ax.legend()
    
    ax.axvline(x=0, color='black', linewidth=0.8)
    
    x_labels = ax.get_xticks()
    ax.set_xticklabels([abs(int(x)) for x in x_labels])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def volcano_plot(results_df, min_fold_change=0.25, max_p_value=0.05, fig_title=None, save_path=None):
    df = results_df.copy()

    df = df[(df['log2FoldChange'] >= -10) & (df['log2FoldChange'] <= 10)]

    df = df[df['padj'] != 0]

    df['log10pval_corrected'] = -np.log10(df['padj'])

    df['significant'] = df['padj'] < max_p_value

    df['outside_range'] = df['significant'] & (df['log2FoldChange'].abs() > min_fold_change)

    df['color'] = 'grey'
    df.loc[df['outside_range'] & (df['log2FoldChange'] > 0), 'color'] = 'red'
    df.loc[df['outside_range'] & (df['log2FoldChange'] <= 0), 'color'] = 'blue'

    plt.figure(figsize=(10, 6))
    plt.scatter(df['log2FoldChange'], df['log10pval_corrected'], s=5, c=df['color'])

    plt.axvline(x=-min_fold_change, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(x=min_fold_change, color='black', linestyle='--', linewidth=0.5)
    plt.axhline(y=-np.log10(max_p_value), color='black', linestyle='--', linewidth=0.5)

    for _, row in df[df['outside_range']].nlargest(20, 'log10pval_corrected').iterrows():
        plt.annotate(row['names'], (row['log2FoldChange'] + 0.2, row['log10pval_corrected']), ha='left', va='center', fontsize=5)

    high_fold_change = df['log2FoldChange'].max()
    print(high_fold_change)
    low_fold_change = df['log2FoldChange'].min()
    print(low_fold_change)
    max_log10_pval = df['log10pval_corrected'].max()

    plt.annotate(f"{df[df['color'] == 'red'].shape[0]} DEGs", xy=(high_fold_change, max_log10_pval), ha='right', va='top', fontsize=10, color='red')
    plt.annotate(f"{df[df['color'] == 'blue'].shape[0]} DEGs", xy=(low_fold_change, max_log10_pval), ha='left', va='top', fontsize=10, color='blue')

    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.2)
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-log10(p-value)')
    plt.xlim(low_fold_change-1, high_fold_change+1)
    if fig_title is not None:
        plt.title(fig_title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def deg_heatmap(adata, sampletag1, sampletag2, fig_tag, results_df, display_top_n=100, min_fold_change=0.25, max_p_value=0.05, save_path=None):
    deg = results_df[(results_df["pvals_adj"] < max_p_value) & (results_df["logfoldchanges"].abs() > min_fold_change)]['names'].tolist()
    
    adata_raw = anndata.AnnData(X=adata.raw.X, obs=adata.obs, var=adata.raw.var)
    
    sc.pp.normalize_total(adata_raw)
    
    mean_expr = adata_raw.to_df().groupby(adata_raw.obs['Sample_Tag']).mean().T
    
    mean_expr = mean_expr[[sampletag1, sampletag2]]
    
    mean_expr['diff'] = mean_expr[sampletag2] - mean_expr[sampletag1]
    
    mean_expr_sorted = mean_expr.sort_values(by='diff', ascending=False)
    
    top_pos = mean_expr_sorted[mean_expr_sorted['diff'] > 0][:display_top_n]
    top_neg = mean_expr_sorted[mean_expr_sorted['diff'] < 0][:display_top_n]
    mean_expr_sorted = pd.concat([top_pos, top_neg])
    
    heatmap_data = mean_expr_sorted.drop(columns='diff')
        
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
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def display_go_enrichment(df, namespace='BP', fig_title=None, save_path=None):
    if df.empty:
        print("Warning: No enriched term.")
        return
    df = df[df['namespace'] == namespace]
    df = df[['GO_term', 'fold_enrichment', 'p_fdr_bh']]
    df['-log10(FDR)'] = -np.log10(df['p_fdr_bh'])
    top_processes = df.nlargest(20, '-log10(FDR)')
    top_processes['GO_term'] = top_processes['GO_term'].apply(lambda x: re.sub(r'\s*\([^)]*\)', '', x))
    top_processes = top_processes.sort_values(by='-log10(FDR)', ascending=False)
    plt.figure(figsize=(10, 8))
    norm = plt.Normalize(top_processes['-log10(FDR)'].min(), top_processes['-log10(FDR)'].max())
    colors = plt.cm.viridis(norm(top_processes['-log10(FDR)']))
    bars = plt.barh(top_processes['GO_term'], top_processes['fold_enrichment'], color=colors)
    
    if fig_title is not None:
        plt.title(fig_title)
    plt.xlabel('Fold Enrichment')
    plt.ylabel(f'{namespace}')
    plt.gca().invert_yaxis()
    
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.ax.set_title('-log10(FDR)', pad=30)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def display_kegg_enrichment(df, fig_title=None, save_path=None):
    if df.empty:
        print("Warning: No enriched term.")
        return
    df = df[['KEGG_term', 'fold_enrichment', 'p_fdr_bh']]
    df['-log10(FDR)'] = -np.log10(df['p_fdr_bh'])
    top_processes = df.nlargest(20, '-log10(FDR)')
    top_processes['KEGG_term'] = top_processes['KEGG_term'].apply(lambda x: re.sub(r'\s*\([^)]*\)', '', x))
    top_processes = top_processes.sort_values(by='-log10(FDR)', ascending=False)
    plt.figure(figsize=(10, 8))
    norm = plt.Normalize(top_processes['-log10(FDR)'].min(), top_processes['-log10(FDR)'].max())
    colors = plt.cm.viridis(norm(top_processes['-log10(FDR)']))
    bars = plt.barh(top_processes['KEGG_term'], top_processes['fold_enrichment'], color=colors)
    
    if fig_title is not None:
        plt.title(fig_title)
    plt.xlabel('Fold Enrichment')
    plt.ylabel('Pathway')
    plt.gca().invert_yaxis()
    
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.ax.set_title('-log10(FDR)', pad=30)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
