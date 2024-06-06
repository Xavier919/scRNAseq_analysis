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

mapping1 = {
 '01 IT-ET Glut': 1,
 '02 NP-CT-L6b Glut': 2,
 '03 OB-CR Glut': 3,
 '04 DG-IMN Glut': 4,
 '05 OB-IMN GABA': 5,
 '06 CTX-CGE GABA': 6,
 '07 CTX-MGE GABA': 7,
 '08 CNU-MGE GABA': 8,
 '09 CNU-LGE GABA': 9,
 '10 LSX GABA': 10,
 '11 CNU-HYa GABA': 11,
 '12 HY GABA': 12,
 '13 CNU-HYa Glut': 13,
 '14 HY Glut': 14,
 '15 HY Gnrh1 Glut': 15,
 '16 HY MM Glut': 16,
 '17 MH-LH Glut': 17,
 '18 TH Glut': 18,
 '19 MB Glut': 19,
 '20 MB GABA': 20,
 '21 MB Dopa': 21,
 '22 MB-HB Sero': 22,
 '23 P Glut': 23,
 '24 MY Glut': 24,
 '25 Pineal Glut': 25,
 '26 P GABA': 26,
 '27 MY GABA': 27,
 '28 CB GABA': 28,
 '29 CB Glut': 29,
 '30 Astro-Epen': 30,
 '31 OPC-Oligo': 31,
 '32 OEC': 32,
 '33 Vascular': 33,
 '34 Immune': 34
}

mapping2 = {'LD_5xFAD': 1,
            'LD_NC': 2,
            'run_5xFAD': 3,
            'run_NC': 4}

mapping2 = {'Multiplet': 1,
            'SampleTag17_flex': 2,
            'SampleTag18_flex': 3,
            'SampleTag19_flex': 4,
            'SampleTag20_flex': 5,
            'Undetermined': 6}
        
def filter_cells_by_gene_counts(adata, min_genes=200, max_genes=6000):
    nonzero_counts = (adata.X > 0).sum(axis=1)
    cell_mask = (nonzero_counts >= min_genes) & (nonzero_counts <= max_genes)
    filtered_adata = adata[cell_mask]
    return filtered_adata

def rm_high_mt(adata, threshold=0):
    mito_genes = [gene for gene in adata.var_names if gene.startswith('mt-')]
    total_counts = adata.X.sum(axis=1).A1 if isinstance(adata.X, np.matrix) else adata.X.sum(axis=1)
    mito_counts = adata[:, mito_genes].X.sum(axis=1).A1 if isinstance(adata.X, np.matrix) else adata[:, mito_genes].X.sum(axis=1)
    mito_percentage = mito_counts / total_counts
    cells_to_keep = mito_percentage <= threshold
    return adata[cells_to_keep, :]

def rm_low_exp(adata, threshold=0.05):
    nonzero_counts = np.array((adata.X != 0).sum(axis=0)).flatten()
    cell_count_threshold = (threshold / 100) * adata.shape[0]
    columns_to_keep = nonzero_counts >= cell_count_threshold
    return adata[:, columns_to_keep]





