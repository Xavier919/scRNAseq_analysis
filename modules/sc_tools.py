import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
from anndata import AnnData
import re
#from goatools.associations import read_ncbi_gene2go
#from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
#from goatools.obo_parser import GODag
#from goatools.anno.genetogo_reader import Gene2GoReader
#from goatools.test_data.genes_NCBI_10090_ProteinCoding import GENEID2NT as MOUSE_GENEID2NT
#import mygene
import pickle
import gseapy as gp
from rpy2.robjects import pandas2ri, r
import rpy2.robjects as ro
from sklearn.utils import resample
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.packages as rpackages
import random
import re
from scipy.sparse import issparse
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects import default_converter
from rpy2.robjects import anndata2ri

def show_pc_variance(adata, layer_name, pc_list=[10,20,50,100]):
    sc.tl.pca(adata, svd_solver='arpack', n_comps=100, use_highly_variable=True, layer=layer_name)
    pca_variance_ratio = adata.uns['pca']['variance_ratio']
    for pc in pc_list:
        print(f'{layer_name} explained variance  for the first {pc} PCs:{np.sum(pca_variance_ratio[:pc])}')

def pearson_normalization(adata):
    analytic_pearson = sc.experimental.pp.normalize_pearson_residuals(adata, inplace=False)
    adata.layers["analytic_pearson_residuals"] = csr_matrix(analytic_pearson["X"])
    
    residuals_sparse = adata.layers["analytic_pearson_residuals"]
    if np.any(np.isnan(residuals_sparse.data)):
        print("Warning: NaNs found in the analytic Pearson residuals.")
        residuals_sparse.data = np.nan_to_num(residuals_sparse.data)
    
    residuals_sum = residuals_sparse.sum(axis=1).A1
    return adata

def scran_normalization(adata):
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