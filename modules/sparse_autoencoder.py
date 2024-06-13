from goatools.associations import read_ncbi_gene2go
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
from goatools.obo_parser import GODag
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.test_data.genes_NCBI_10090_ProteinCoding import GENEID2NT as MOUSE_GENEID2NT
import mygene
from typing import Callable, Any
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Any, Union, Dict
import umap
import matplotlib.pyplot as plt
import pandas as pd
import anndata
import scanpy as sc
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def LN(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std

class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias

class TopK(nn.Module):
    def __init__(self, k: int, postact_fn: Callable = nn.ReLU()) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict.update({prefix + "k": self.k, prefix + "postact_fn": self.postact_fn.__class__.__name__})
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> "TopK":
        k = state_dict["k"]
        postact_fn = ACTIVATIONS_CLASSES[state_dict["postact_fn"]]()
        return cls(k=k, postact_fn=postact_fn)

ACTIVATIONS_CLASSES = {
    "ReLU": nn.ReLU,
    "Identity": nn.Identity,
    "TopK": TopK,
}

class Autoencoder(nn.Module):
    """Sparse autoencoder

    Implements:
        latents = activation(encoder(x - pre_bias) + latent_bias)
        recons = decoder(latents) + pre_bias
    """

    def __init__(
        self, n_latents: int, n_inputs: int, activation: Callable = nn.ReLU(), tied: bool = False,
        normalize: bool = False
    ) -> None:
        """
        :param n_latents: dimension of the autoencoder latent
        :param n_inputs: dimensionality of the original data (e.g residual stream, number of MLP hidden units)
        :param activation: activation function
        :param tied: whether to tie the encoder and decoder weights
        """
        super().__init__()

        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))
        self.encoder = nn.Linear(n_inputs, n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.activation = activation
        if tied:
            self.decoder = TiedTranspose(self.encoder)
        else:
            self.decoder = nn.Linear(n_latents, n_inputs, bias=False)
        self.normalize = normalize

        self.stats_last_nonzero: torch.Tensor
        self.latents_activation_frequency: torch.Tensor
        self.latents_mean_square: torch.Tensor
        self.register_buffer("stats_last_nonzero", torch.zeros(n_latents, dtype=torch.long))
        self.register_buffer("latents_activation_frequency", torch.ones(n_latents, dtype=torch.float))
        self.register_buffer("latents_mean_square", torch.zeros(n_latents, dtype=torch.float))

    def encode_pre_act(self, x: torch.Tensor, latent_slice: slice = slice(None)) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, n_inputs])
        :param latent_slice: slice of latents to compute
            Example: latent_slice = slice(0, 10) to compute only the first 10 latents.
        :return: autoencoder latents before activation (shape: [batch, n_latents])
        """
        x = x - self.pre_bias
        latents_pre_act = F.linear(
            x, self.encoder.weight[latent_slice], self.latent_bias[latent_slice]
        )
        return latents_pre_act

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, Dict[str, Any]]:
        if not self.normalize:
            return x, dict()
        x, mu, std = LN(x)
        return x, dict(mu=mu, std=std)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return: autoencoder latents (shape: [batch, n_latents])
        """
        x, info = self.preprocess(x)
        return self.activation(self.encode_pre_act(x)), info

    def decode(self, latents: torch.Tensor, info: Union[Dict[str, Any], None] = None) -> torch.Tensor:
        """
        :param latents: autoencoder latents (shape: [batch, n_latents])
        :return: reconstructed data (shape: [batch, n_inputs])
        """
        ret = self.decoder(latents) + self.pre_bias
        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return:  autoencoder latents pre activation (shape: [batch, n_latents])
                  autoencoder latents (shape: [batch, n_latents])
                  reconstructed data (shape: [batch, n_inputs])
        """
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(latents_pre_act)
        recons = self.decode(latents, info)

        # set all indices of self.stats_last_nonzero where (latents != 0) to 0
        self.stats_last_nonzero *= (latents == 0).all(dim=0).long()
        self.stats_last_nonzero += 1

        return latents_pre_act, latents, recons

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> "Autoencoder":
        n_latents, d_model = state_dict["encoder.weight"].shape

        # Retrieve activation
        activation_class_name = state_dict.pop("activation", "ReLU")
        activation_class = ACTIVATIONS_CLASSES.get(activation_class_name, nn.ReLU)
        normalize = activation_class_name == "TopK"  # NOTE: hacky way to determine if normalization is enabled
        activation_state_dict = state_dict.pop("activation_state_dict", {})
        if hasattr(activation_class, "from_state_dict"):
            activation = activation_class.from_state_dict(
                activation_state_dict, strict=strict
            )
        else:
            activation = activation_class()
            if hasattr(activation, "load_state_dict"):
                activation.load_state_dict(activation_state_dict, strict=strict)

        autoencoder = cls(n_latents, d_model, activation=activation, normalize=normalize)
        # Load remaining state dict
        autoencoder.load_state_dict(state_dict, strict=strict)
        return autoencoder

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        sd[prefix + "activation"] = self.activation.__class__.__name__
        if hasattr(self.activation, "state_dict"):
            sd[prefix + "activation_state_dict"] = self.activation.state_dict()
        return sd


def save_model(autoencoder: nn.Module, path: str) -> None:
    """
    Saves the autoencoder model state to the specified path.

    :param autoencoder: The trained autoencoder model
    :param path: The file path to save the model state
    """
    state = {
        'model_state': autoencoder.state_dict()
    }
    torch.save(state, path)
    print(f"Model state saved to {path}")

def load_model(autoencoder: nn.Module, path: str) -> None:
    """
    Loads the autoencoder model state from the specified path.

    :param autoencoder: The autoencoder model instance
    :param path: The file path from which to load the model state
    """
    state = torch.load(path)
    model_state = state['model_state']

    # Handle activation state dict if it exists
    if 'activation_state_dict' in model_state:
        activation_state_dict = model_state.pop('activation_state_dict')
        if hasattr(autoencoder.activation, 'load_state_dict'):
            autoencoder.activation.load_state_dict(activation_state_dict)

    autoencoder.load_state_dict(model_state)
    print(f"Model state loaded from {path}")


# Define a custom dataset class for AnnData
class AnnDataDataset(Dataset):
    def __init__(self, adata):
        self.adata = adata

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        # Convert data to a tensor and ensure the label is a compatible type
        x = torch.tensor(self.adata.X[idx].toarray(), dtype=torch.float32)
        return x


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


def plot_latent_heatmap(autoencoder, data, adata, sample_tags=None, feature_range=(0, 1), num_cells=500):
    """
    Plots a heatmap of latent activations, optionally displaying only specific sample tags.
    
    Parameters:
    - autoencoder: Trained autoencoder model.
    - data: Input data to be encoded.
    - adata: AnnData object containing the observations.
    - sample_tags: List of specific sample tags to display. If None, displays all.
    - feature_range: Desired range of transformed data.
    - num_cells: Number of cells to display in the heatmap. Default is 500.
    """
    # Encode the data to get latent representations
    autoencoder.eval()  # Ensure the autoencoder is in evaluation mode
    with torch.no_grad():
        latents, _ = autoencoder.encode(data)
    latents_np = latents.detach().cpu().numpy()
    sample_tags_all = adata.obs['Sample_Tag']
    
    # Scale each latent dimension separately to the range [0, 1]
    scaler = MinMaxScaler(feature_range=feature_range)
    latents_np_scaled = scaler.fit_transform(latents_np)
    
    # Filter sample tags if provided
    if sample_tags:
        mask_sample_tags = sample_tags_all.isin(sample_tags)
        latents_np_scaled = latents_np_scaled[mask_sample_tags]
    
    # Randomly sample cells if there are more than num_cells
    if latents_np_scaled.shape[0] > num_cells:
        indices = np.random.choice(latents_np_scaled.shape[0], num_cells, replace=False)
        latents_np_scaled = latents_np_scaled[indices]
    
    # Verification of scaling
    min_values = np.min(latents_np_scaled, axis=0)
    max_values = np.max(latents_np_scaled, axis=0)
    #print(f'Min values after scaling: {min_values}')
    #print(f'Max values after scaling: {max_values}')
    
    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(latents_np_scaled, cmap='viridis', yticklabels=False)
    plt.xlabel('Latent dimensions')
    plt.ylabel('Cells')
    plt.title('Heatmap of latent activations')
    plt.show()


def plot_umap_embedding(embedding, adata, sample_tags=None, min_count=100):
    """
    Plots UMAP embedding, optionally displaying only specific sample tags.
    The legend will always display cell types.
    Parameters:
    - embedding: numpy array of UMAP embeddings.
    - adata: AnnData object containing the observations.
    - sample_tags: List of specific sample tags to display. If None, displays all.
    - min_count: Minimum number of observations for a cell type to be displayed.
    """
    cell_types = adata.obs['class_name']
    sample_tags_all = adata.obs['Sample_Tag']
    # Filter cell types by min_count
    cell_type_counts = cell_types.value_counts()
    valid_cell_types = cell_type_counts[cell_type_counts >= min_count].index
    mask_cell_types = cell_types.isin(valid_cell_types)
    filtered_embedding = embedding[mask_cell_types]
    filtered_cell_types = cell_types[mask_cell_types]
    filtered_sample_tags = sample_tags_all[mask_cell_types]
    # Filter sample tags if provided
    if sample_tags:
        mask_sample_tags = filtered_sample_tags.isin(sample_tags)
        final_embedding = filtered_embedding[mask_sample_tags]
        final_cell_types = filtered_cell_types[mask_sample_tags]
    else:
        final_embedding = filtered_embedding
        final_cell_types = filtered_cell_types
    # Create a consistent color palette
    unique_cell_types = valid_cell_types
    palette = sns.color_palette('tab10', len(unique_cell_types))
    color_dict = {cell_type: palette[i] for i, cell_type in enumerate(unique_cell_types)}
    # Plot UMAP embedding
    plt.figure(figsize=(6, 4))
    scatter = sns.scatterplot(x=final_embedding[:, 0], y=final_embedding[:, 1], hue=final_cell_types, palette=color_dict, s=5)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('UMAP of latent representations')
    # Adjust legend to be outside the plot
    handles, labels = scatter.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5), markerscale=3)
    plt.tight_layout()
    plt.show()

def plot_activation_frequency(autoencoder: nn.Module) -> None:
    """
    Plots the activation frequency of latent units in the autoencoder.

    :param autoencoder: The trained autoencoder model
    """
    # Get the activation frequency from the model
    activation_freq = autoencoder.latents_activation_frequency.detach().numpy()

    # Plot the activation frequency
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(activation_freq)), activation_freq)
    plt.xlabel('Latent Dimensions')
    plt.ylabel('Activation Frequency')
    plt.title('Activation Frequency of Latent Units')
    plt.show()

def plot_cell_reconstruction(autoencoder, data, cell_idx=0):
    """
    Plots original vs. reconstructed data for a specific cell.
    
    Parameters:
    - autoencoder: Trained autoencoder model.
    - data: Input data to be encoded.
    - cell_idx: Index of the cell to visualize. Default is 0.
    """

    # Get reconstructed data
    _, _, recons = autoencoder(data)

    # Convert to NumPy arrays for visualization
    original_data = data.numpy()
    reconstructed_data = recons.detach().numpy()

    # Plot original vs. reconstructed data for the specified cell
    plt.figure(figsize=(10, 5))
    plt.plot(original_data[cell_idx, :], label='Original')
    plt.plot(reconstructed_data[cell_idx, :], label='Reconstructed')
    plt.xlabel('Genes')
    plt.ylabel('Expression Level')
    plt.title(f'Original vs. reconstructed data for cell {cell_idx}')
    plt.legend()
    plt.show()

def plot_gene_reconstruction(autoencoder, data, gene_idx=0):
    """
    Plots original vs. reconstructed data for a specific gene.
    
    Parameters:
    - autoencoder: Trained autoencoder model.
    - data: Input data to be encoded.
    - gene_idx: Index of the gene to visualize. Default is 0.
    """
    # Get reconstructed data
    _, _, recons = autoencoder(data)

    # Convert to NumPy arrays for visualization
    original_data = data.numpy()
    reconstructed_data = recons.detach().numpy()

    # Plot original vs. reconstructed data for the specified gene
    plt.figure(figsize=(10, 5))
    plt.plot(original_data[:, gene_idx], label='Original')
    plt.plot(reconstructed_data[:, gene_idx], label='Reconstructed')
    plt.xlabel('Cells')
    plt.ylabel('Expression Level')
    plt.title(f'Original vs. reconstructed data for gene {gene_idx}')
    plt.legend()
    plt.show()


def plot_top_contributing_genes(autoencoder, adata, latent_dim=0, top_n=10):
    """
    Plot the top contributing genes for a specific latent dimension of the autoencoder.

    Parameters:
    autoencoder (nn.Module): The trained autoencoder model.
    adata (anndata.AnnData): The annotated data matrix containing gene names.
    latent_dim (int): The latent dimension to visualize. Default is 0.
    top_n (int): The number of top contributing genes to plot. Default is 10.
    """
    # Extract encoder weights
    encoder_weights = autoencoder.encoder.weight.detach().numpy()
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame(encoder_weights.T, index=adata.var_names)
    
    # Plot top contributing genes for the specified latent dimension
    top_genes = df.iloc[:, latent_dim].sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, 5))
    top_genes.plot(kind='bar')
    plt.xlabel('Genes')
    plt.ylabel('Contribution weight')
    plt.title(f'Top contributing genes for latent dimension {latent_dim}')
    plt.show()