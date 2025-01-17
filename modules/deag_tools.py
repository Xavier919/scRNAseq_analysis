import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import anndata
from anndata import AnnData
import gseapy as gp
from rpy2.robjects import pandas2ri
from sklearn.utils import resample

import matplotlib.pyplot as plt


def get_DEGs(df, max_pval=0.05, min_fold_change=0.25):
    """
    Get differentially expressed genes (DEGs) from a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing gene expression data.
        max_pval (float, optional): The maximum p-value threshold for statistical significance. Default is 0.05.
        min_fold_change (float, optional): The minimum fold change threshold for differential expression. Default is 0.25.

    Returns:
        tuple: A tuple containing two lists of DEGs. The first list contains genes with fold change above the threshold,
               and the second list contains genes with fold change below the negative threshold.
    """
    filtered_above = df[(df['padj'] < max_pval) & (df['log2FoldChange'] > min_fold_change)]
    filtered_below = df[(df['padj'] < max_pval) & (df['log2FoldChange'] < -min_fold_change)]
    genes_above_name = [x.upper() for x in filtered_above['names']]
    genes_below_name = [x.upper() for x in filtered_below['names']]
    return genes_above_name, genes_below_name

def query_genes(adata, save_path=None):
    """
    Query genes using MyGeneInfo API and retrieve their corresponding NCBI gene IDs.

    Parameters:
        adata (AnnData): Annotated data object containing gene expression data.
        save_path (str, optional): Path to save the gene-to-NCBI dictionary as a pickle file.

    Returns:
        dict: A dictionary mapping gene symbols to their corresponding NCBI gene IDs.
    """
    reference_list = adata.var.index.str.upper().tolist()
    mg = mygene.MyGeneInfo()
    background_gene_info = mg.querymany(reference_list, scopes='symbol', fields='entrezgene', species='mouse')
    gene_to_ncbi = {entry['query']: int(entry.get('entrezgene')) for entry in background_gene_info if 'entrezgene' in entry}
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(gene_to_ncbi, f)
    return gene_to_ncbi

def go_enrichment_analysis(gene_list, save_path=None):
    """
    Perform Gene Ontology (GO) enrichment analysis on a given gene list.

    Args:
        gene_list (list): A list of genes to be analyzed.
        save_path (str, optional): The file path to save the results. Defaults to None.

    Returns:
        pandas.DataFrame: A DataFrame containing the enriched GO terms and associated statistics.

    """
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

        if results.empty:
            continue
        
        results_sig = results[results['Adjusted P-value'] < 0.05]

        if results_sig.empty:
            continue
        
        # Assuming a fixed population size for simplicity
        population_size = 21836  
        study_size = len(gene_list)

        study_counts = results_sig['Overlap'].apply(lambda x: int(x.split('/')[0]))
        population_counts = results_sig['Overlap'].apply(lambda x: int(x.split('/')[1]))

        fold_enrichment = (study_counts / study_size) / (population_counts / population_size)

        df_results = pd.DataFrame({
            'GO_term': results_sig['Term'],
            'study_count': study_counts,
            'population_count': population_counts,
            'study_items': results_sig['Genes'].str.split(';'),
            'p_uncorrected': results_sig['P-value'],
            'p_fdr_bh': results_sig['Adjusted P-value'],
            'fold_enrichment': fold_enrichment,
            'enrichment': results_sig['Combined Score'].apply(lambda x: 'enriched' if x > 1 else 'purified'),
            'namespace': namespace
        })
        
        all_results.append(df_results)
    
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
    else:
        final_results = pd.DataFrame()

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(final_results, f)

    return final_results

def kegg_enrichment_analysis(gene_list, save_path=None):
    """
    Perform KEGG enrichment analysis on a given gene list.

    Parameters:
    - gene_list (list): A list of genes to be analyzed.
    - save_path (str, optional): The path to save the results as a pickle file. Default is None.

    Returns:
    - df_results (DataFrame): A pandas DataFrame containing the results of the enrichment analysis.

    """
    enr = gp.enrichr(gene_list=gene_list,
                     gene_sets='KEGG_2019_Mouse',
                     outdir=None, 
                     cutoff=0.05)  

    results = enr.results

    results_sig = results[results['Adjusted P-value'] < 0.05]

    population_size = 21836 
    study_size = len(gene_list)
    study_counts = results_sig['Overlap'].apply(lambda x: int(x.split('/')[0]))
    population_counts = results_sig['Overlap'].apply(lambda x: int(x.split('/')[1]))
    fold_enrichment = (study_counts / study_size) / (population_counts / population_size)

    df_results = pd.DataFrame({
        'KEGG_term': results_sig['Term'],
        'study_count': results_sig['Overlap'].apply(lambda x: int(x.split('/')[0])),
        'population_count': results_sig['Overlap'].apply(lambda x: int(x.split('/')[1])),
        'study_items': results_sig['Genes'].str.split(';'),
        'p_uncorrected': results_sig['P-value'],
        'p_fdr_bh': results_sig['Adjusted P-value'],
        'fold_enrichment': fold_enrichment,
        'enrichment': results_sig['Combined Score'].apply(lambda x: 'enriched' if x > 1 else 'purified')
    })

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(df_results, f)

    return df_results

def wilcoxon_dea(adata, save_path=None):
    """
    Perform differential expression analysis using the Wilcoxon rank-sum test.

    Parameters:
        adata (AnnData): Annotated data object containing gene expression data.
        save_path (str, optional): Path to save the results. If provided, the results will be saved as a pickle file.

    Returns:
        pandas.DataFrame: DataFrame containing the differential expression results.

    """
    adata.obs['group'] = adata.obs['group'].astype('category')
    sc.tl.rank_genes_groups(adata, groupby='group', reference='control', method='wilcoxon', corr_method='benjamini-hochberg', layer='log1p_norm')
    de_results_df = sc.get.rank_genes_groups_df(adata, group='condition')
    de_results_df.rename(columns={'logfoldchanges': 'log2FoldChange', 'pvals_adj': 'padj'}, inplace=True)
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(save_path, f)
    return de_results_df

def aggregate_and_filter(
    adata,
    cell_identity,
    condition_key="Sample_Tag",
    cell_identity_key="class_name",
    obs_to_keep=[], 
    replicates_per_condition=3,
    seed=42
):
    """
    Aggregate and filter the data in the AnnData object based on specified parameters.

    Parameters:
        adata (AnnData): Annotated data matrix with observations (cells) in rows and variables (genes) in columns.
        cell_identity (str): The cell identity to filter the data for.
        condition_key (str, optional): The key in `adata.obs` that represents the condition. Defaults to "Sample_Tag".
        cell_identity_key (str, optional): The key in `adata.obs` that represents the cell identity. Defaults to "class_name".
        obs_to_keep (list, optional): Additional observation columns to keep in the aggregated data. Defaults to an empty list.
        replicates_per_condition (int, optional): The number of replicates per condition to include in the aggregated data. Defaults to 3.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        AnnData: Annotated data matrix with aggregated and filtered data.

    """
    random.seed(seed)
    np.random.seed(seed)
    
    adata_cell_pop = adata[adata.obs[cell_identity_key] == cell_identity].copy()
    
    condition_counts = adata_cell_pop.obs[condition_key].value_counts()
    conditions_to_drop = condition_counts[condition_counts <= 100].index.tolist()

    df = pd.DataFrame(columns=[*adata_cell_pop.var_names, *obs_to_keep])

    adata_cell_pop.obs[condition_key] = adata_cell_pop.obs[condition_key].astype("category")
    
    for condition in adata_cell_pop.obs[condition_key].cat.categories:
        if condition in conditions_to_drop:
            continue
        
        adata_condition = adata_cell_pop[adata_cell_pop.obs[condition_key] == condition]
        indices = np.array_split(random.sample(list(adata_condition.obs_names), len(adata_condition)), replicates_per_condition)
        
        for i, rep_idx in enumerate(indices):
            adata_replicate = adata_condition[rep_idx]
            
            agg_dict = {gene: "sum" for gene in adata_replicate.var_names}
            agg_dict.update({obs: "first" for obs in obs_to_keep})
            
            df_condition = pd.DataFrame(adata_replicate.X.toarray(), index=adata_replicate.obs_names, columns=adata_replicate.var_names)
            df_condition = df_condition.join(adata_replicate.obs[obs_to_keep])
            df_condition = df_condition.groupby(condition_key).agg(agg_dict)
            df_condition[condition_key] = condition
            
            new_index = f"{condition}_{i}"
            df.loc[new_index] = df_condition.loc[condition]
    
    return sc.AnnData(df[adata_cell_pop.var_names], obs=df.drop(columns=adata_cell_pop.var_names))


def deseq2_dea(control_df, condition_df, save_path):
    """
    Perform differential expression analysis using DESeq2.

    Args:
        control_df (pd.DataFrame): DataFrame containing the control samples' count data.
        condition_df (pd.DataFrame): DataFrame containing the condition samples' count data.
        save_path (str): Path to save the results as a pickle file. If None, the results will not be saved.

    Returns:
        pd.DataFrame: DataFrame containing the differential expression analysis results.

    """
    merged_df = pd.concat([control_df, condition_df]).T
    metadata = pd.DataFrame(
        [(col, col.split('_')[0], "control" if col in control_df.index else "condition")
        for col in merged_df.columns],
        columns=["Replicates", "Sample_Tag", "Condition"]
    ).set_index("Replicates")
    r_counts = pandas2ri.py2rpy(merged_df)
    r_metadata = pandas2ri.py2rpy(metadata)
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
        res <- results(dds, contrast=c("Condition", "condition", "control"))
        res_df <- as.data.frame(res)
        return(res_df)
    }
    """
    pandas2ri.activate()
    
    ro.r(deseq2_script)
    
    run_deseq2 = ro.globalenv['run_deseq2']
    res_r = run_deseq2(r_counts, r_metadata)
    with localconverter(ro.default_converter + pandas2ri.converter):
        de_results_df = ro.conversion.rpy2py(res_r)
        
    de_results_df = de_results_df.reset_index().rename(columns={'index': 'names'})

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(de_results_df, f)
    return de_results_df

def mast_dea(adata, control_df, condition_df, save_path=None):
    """
    Perform MAST (Model-based Analysis of Single-cell Transcriptomics) analysis for differential expression analysis.

    Parameters:
        adata (AnnData): Annotated data object containing single-cell transcriptomics data.
        control_df (pandas.DataFrame): DataFrame containing control group data.
        condition_df (pandas.DataFrame): DataFrame containing condition group data.
        save_path (str, optional): Path to save the results as a pickle file. Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame containing the results of differential expression analysis.

    Raises:
        None

    Example usage:
        # Load data
        adata = load_data()
        control_df = load_control_data()
        condition_df = load_condition_data()

        # Perform MAST analysis
        results = mast_dea(adata, control_df, condition_df, save_path='results.pkl')
    """
    control_indexes = control_df.index
    condition_indexes = condition_df.index
    cdata1 = pd.DataFrame({'wellKey': control_indexes, 'group': 'control'})
    cdata2 = pd.DataFrame({'wellKey': condition_indexes, 'group': 'condition'})
    cdata = pd.concat([cdata1, cdata2], ignore_index=True)
    cdata.set_index('wellKey', inplace=True)
    exprs_data = pd.concat([control_df, condition_df]).T
    fdata = pd.DataFrame(index=adata.var_names)
    fdata['primerid'] = fdata.index
    fdata.index.name = 'primerid'
    # Convert to R dataframes using the correct localconverter
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_exprs_data = ro.conversion.py2rpy(exprs_data)
        r_cdata = ro.conversion.py2rpy(cdata)
        r_fdata = ro.conversion.py2rpy(fdata)
    
    # Assign R objects directly
    ro.globalenv['exprs_data'] = r_exprs_data
    ro.globalenv['cdata'] = r_cdata
    ro.globalenv['fdata'] = r_fdata
    
    # MAST analysis script with debug prints
    mast_script = """
    library(MAST)
    
    run_mast <- function(exprs_data, cdata, fdata) {
        print(paste("Shape of exprs_data:", paste(dim(exprs_data), collapse=" x ")))
        print(paste("Shape of cdata:", paste(dim(cdata), collapse=" x ")))
        print(paste("Shape of fdata:", paste(dim(fdata), collapse=" x ")))
        
        exprs_data <- as.matrix(exprs_data)
        cdata <- as.data.frame(cdata)
        fdata <- as.data.frame(fdata)
        
        # Ensure group is a factor and set reference level
        cdata$group <- factor(cdata$group, levels = c('control', 'condition'))
        
        sca <- FromMatrix(exprsArray=exprs_data, cData=cdata, fData=fdata, check_sanity=FALSE)
        zlmCond <- zlm(~ group, sca)
        summaryCond <- summary(zlmCond, doLRT='groupcondition')
        summaryDt <- summaryCond$datatable
        fcHurdle <- merge(summaryDt[contrast=='groupcondition' & component=='H', .(primerid, `Pr(>Chisq)`)], 
                          summaryDt[contrast=='groupcondition' & component=='logFC', .(primerid, coef)], 
                          by='primerid')
        fcHurdle[, fdr := p.adjust(`Pr(>Chisq)`, 'fdr')]
        return(fcHurdle)
    }
    """
    ro.r(mast_script)
    
    # Run MAST analysis
    run_mast = ro.globalenv['run_mast']
    res_r = run_mast(ro.globalenv['exprs_data'], ro.globalenv['cdata'], ro.globalenv['fdata'])
    
    # Convert results back to a Pandas dataframe using the correct localconverter
    with localconverter(ro.default_converter + pandas2ri.converter):
        de_results_df = ro.conversion.rpy2py(res_r)
    
    # Rename columns
    de_results_df.rename(columns={
        'primerid': 'names',
        'coef': 'log2FoldChange',
        'fdr': 'padj'
    }, inplace=True)

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(de_results_df, f)
    return de_results_df

def horizontal_deg_chart(adata, min_fold_change=0.25, max_p_value=0.05, n_subsamples=5, fig_title=None, save_path=None):
    """
    Generate a horizontal bar chart showing the number of positive and negative differentially expressed genes (DEGs) 
    for each cluster in the given AnnData object.

    Parameters:
        adata (AnnData): Annotated data object containing gene expression data.
        min_fold_change (float, optional): Minimum fold change threshold for DEGs. Default is 0.25.
        max_p_value (float, optional): Maximum p-value threshold for DEGs. Default is 0.05.
        n_subsamples (int, optional): Number of subsamples to use for DEG analysis. Default is 5.
        fig_title (str, optional): Title for the chart. Default is None.
        save_path (str, optional): File path to save the chart. Default is None.

    Returns:
        None
    """
    cluster_n_DEGs = []

    cluster_class_names = adata.obs['cluster_class_name'].unique()

    for cluster_class_name in tqdm(cluster_class_names):
        subset_adata = adata[adata.obs['cluster_class_name'] == cluster_class_name, :].copy()
        subset_adata = subset_adata[subset_adata.obs['group'] != "undefined", :]
        df = DEG_analysis(subset_adata)
        if df is None:
            continue
        positive_enriched = df[(df['log2FoldChange'] > min_fold_change) & (df['padj'] < max_p_value)]
        negative_enriched = df[(df['log2FoldChange'] < -min_fold_change) & (df['padj'] < max_p_value)]
        positive_count = positive_enriched.shape[0]
        negative_count = negative_enriched.shape[0]
        cluster_n_DEGs.append((cluster_class_name, positive_count, negative_count))
    
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
    """
    Generate a volcano plot based on the results dataframe.

    Parameters:
    - results_df (pandas.DataFrame): The dataframe containing the results.
    - min_fold_change (float, optional): The minimum fold change threshold. Default is 0.25.
    - max_p_value (float, optional): The maximum p-value threshold. Default is 0.05.
    - fig_title (str, optional): The title of the plot. Default is None.
    - save_path (str, optional): The file path to save the plot. Default is None.

    Returns:
    None
    """
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

    top_genes = df[df['outside_range']].nlargest(20, 'log10pval_corrected')
    for _, row in top_genes.iterrows():
        plt.annotate(row['names'], (row['log2FoldChange'] + 0.2, row['log10pval_corrected']), ha='left', va='center', fontsize=5)

    high_fold_change = df['log2FoldChange'].max()
    low_fold_change = df['log2FoldChange'].min()
    max_log10_pval = df['log10pval_corrected'].max()

    plt.annotate(f"{df[df['color'] == 'red'].shape[0]} DEGs", xy=(high_fold_change+0.75, max_log10_pval), ha='right', va='top', fontsize=10, color='red')
    plt.annotate(f"{df[df['color'] == 'blue'].shape[0]} DEGs", xy=(low_fold_change-0.75, max_log10_pval), ha='left', va='top', fontsize=10, color='blue')

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
    """
    Generate a differential expression heatmap.

    Parameters:
        adata (AnnData): Annotated data object containing gene expression data.
        sampletag1 (str): Name of the first sample tag.
        sampletag2 (str): Name of the second sample tag.
        fig_tag (str): Tag for the generated figure.
        results_df (DataFrame): DataFrame containing differential expression results.
        display_top_n (int, optional): Number of top differentially expressed genes to display. Defaults to 100.
        min_fold_change (float, optional): Minimum fold change threshold for differential expression. Defaults to 0.25.
        max_p_value (float, optional): Maximum p-value threshold for differential expression. Defaults to 0.05.
        save_path (str, optional): File path to save the generated heatmap. Defaults to None.

    Returns:
        None
    """
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
    """
    Display the Gene Ontology (GO) enrichment results.

    Args:
        df (pandas.DataFrame): The dataframe containing the GO enrichment results.
        namespace (str, optional): The GO namespace to display (default is 'BP' for Biological Process).
        fig_title (str, optional): The title of the figure (default is None).
        save_path (str, optional): The file path to save the figure (default is None).

    Returns:
        None

    Raises:
        None

    """
    if df.empty:
        print("Warning: No enriched term.")
        return

    df = df[df['namespace'] == namespace]
    df = df[['GO_term', 'p_fdr_bh']]
    df['-log10(FDR)'] = -np.log10(df['p_fdr_bh'])
    top_processes = df.nlargest(20, '-log10(FDR)')
    top_processes['GO_term'] = top_processes['GO_term'].apply(lambda x: re.sub(r'\s*\([^)]*\)', '', x))
    top_processes = top_processes.sort_values(by='-log10(FDR)', ascending=False)
    
    plt.figure(figsize=(8, 6))
    bars = plt.barh(top_processes['GO_term'], top_processes['-log10(FDR)'], color='skyblue')
    
    if fig_title is not None:
        plt.title(fig_title)
    plt.xlabel('-log10(FDR)')
    plt.ylabel(f'{namespace}')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def display_kegg_enrichment(df, fig_title=None, save_path=None):
    """
    Display KEGG enrichment results as a horizontal bar plot.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing KEGG enrichment results.
    - fig_title (str, optional): The title of the figure. Default is None.
    - save_path (str, optional): The file path to save the figure. Default is None.

    Returns:
    None
    """
    if df.empty:
        print("Warning: No enriched term.")
        return

    df = df[['KEGG_term', 'p_fdr_bh']]
    df['-log10(FDR)'] = -np.log10(df['p_fdr_bh'])
    top_processes = df.nlargest(20, '-log10(FDR)')
    top_processes['KEGG_term'] = top_processes['KEGG_term'].apply(lambda x: re.sub(r'\s*\([^)]*\)', '', x))
    top_processes = top_processes.sort_values(by='-log10(FDR)', ascending=False)
    
    plt.figure(figsize=(8, 6))
    bars = plt.barh(top_processes['KEGG_term'], top_processes['-log10(FDR)'], color='skyblue')
    
    if fig_title is not None:
        plt.title(fig_title)
    plt.xlabel('-log10(FDR)')
    plt.ylabel('Pathway')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
