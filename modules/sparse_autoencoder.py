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
import torch.nn.utils.prune as prune
from typing import Callable, Dict, Any, Union
from tqdm import tqdm

def apply_pruning(model, amount=0.5):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == model.encoder.in_features:
            weight = module.weight.data
            n_latents = weight.size(0)
            for i in range(n_latents):
                w = weight[i, :]
                threshold = torch.quantile(w.abs(), amount)
                mask = w.abs() >= threshold
                w *= mask.float()

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

# Function to evaluate the model on the test set
def evaluate_autoencoder(autoencoder, dataloader, loss_fn, device):
    autoencoder.eval()
    test_loss = 0
    with torch.no_grad():
        for data in dataloader:
            inputs = data[0].to(device)
            _, latents, outputs = autoencoder(inputs)
            loss = loss_fn(outputs, inputs)
            test_loss += loss.item()
    return test_loss / len(dataloader)

# Example training loop with pruning and test set evaluation
def train_autoencoder(autoencoder, train_loader, test_loader, device, num_epochs=50, learning_rate=0.0001, prune_interval=5, prune_amount=0.5):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    # Variable to keep track of the best test loss
    best_test_loss = float('inf')
    
    for epoch in range(num_epochs):
        autoencoder.train()
        for data in tqdm(train_loader):
            inputs = data[0].to(device)

            # Forward pass
            _, latents, outputs = autoencoder(inputs)
            loss = loss_fn(outputs, inputs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Apply pruning at specified intervals
        if (epoch + 1) % prune_interval == 0:
            apply_pruning(autoencoder, amount=prune_amount)
            print(f"Pruning applied at epoch {epoch + 1}")

        # Evaluate on test set
        test_loss = evaluate_autoencoder(autoencoder, test_loader, loss_fn, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Test Loss: {test_loss}")

        # Check if the current test loss is the best we've seen so far
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            # Save the model
            torch.save(autoencoder.state_dict(), 'best_autoencoder.pth')
            print(f"Saved new best model with test loss: {best_test_loss}")


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


def plot_latent_heatmap(autoencoder, data, adata, sample_tags=None, clusters=None, subclusters=None, num_cells=500):
    """
    Plots a heatmap of latent activations, optionally displaying only specific sample tags or a specific cluster.
    
    Parameters:
    - autoencoder: Trained autoencoder model.
    - data: Input data to be encoded.
    - adata: AnnData object containing the observations.
    - sample_tags: List of specific sample tags to display. If None, displays all.
    - cluster: Specific cluster to display. If None, displays all.
    - feature_range: Desired range of transformed data.
    - num_cells: Number of cells to display in the heatmap. Default is 500.
    """
    # Encode the data to get latent representations
    autoencoder.eval()  # Ensure the autoencoder is in evaluation mode
    with torch.no_grad():
        latents, _ = autoencoder.encode(data)
    latents_np = latents.detach().cpu().numpy()
    
    # Scale each latent dimension separately to the range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    latents_np_scaled = scaler.fit_transform(latents_np)
    
    # Filter by sample tags and cluster if provided
    mask = np.ones(len(adata.obs), dtype=bool)
    if sample_tags:
        mask &= adata.obs['Sample_Tag'].isin(sample_tags)
    if clusters:
        mask &= adata.obs['cluster_class_name'].isin(clusters)
    if subclusters:
        mask &= adata.obs['cluster_subclass_name'].isin(subclusters)
    
    latents_np_scaled = latents_np_scaled[mask]
    
    # Randomly sample cells if there are more than num_cells
    if latents_np_scaled.shape[0] > num_cells:
        indices = np.random.choice(latents_np_scaled.shape[0], num_cells, replace=False)
        latents_np_scaled = latents_np_scaled[indices]
    
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
    Plot the top positive and negative contributing genes for a specific latent dimension of the autoencoder.

    Parameters:
    autoencoder (nn.Module): The trained autoencoder model.
    adata (anndata.AnnData): The annotated data matrix containing gene names.
    latent_dim (int): The latent dimension to visualize. Default is 0.
    top_n (int): The number of top contributing genes to plot for both positive and negative contributions. Default is 10.
    """
    # Extract encoder weights
    encoder_weights = autoencoder.encoder.weight.detach().numpy()
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame(encoder_weights.T, index=adata.var_names)
    
    # Get top positive contributing genes
    top_positive_genes = df.iloc[:, latent_dim].sort_values(ascending=False).head(top_n)
    
    # Get top negative contributing genes
    top_negative_genes = df.iloc[:, latent_dim].sort_values(ascending=True).head(top_n)
    
    # Concatenate top positive and negative genes
    top_genes = pd.concat([top_positive_genes, top_negative_genes])
    
    # Plot
    plt.figure(figsize=(10, 6))
    top_genes.plot(kind='bar', color=['blue' if w > 0 else 'red' for w in top_genes])
    plt.xlabel('Genes')
    plt.ylabel('Contribution weight')
    plt.title(f'Top contributing genes for latent dimension {latent_dim}')
    plt.savefig(f'figures/top_contributing_{latent_dim}')
    plt.show()


def get_top_genes(autoencoder, adata, top_n=10):
    """
    Get the top positive and negative contributing genes for each latent dimension of the autoencoder.

    Parameters:
    autoencoder (nn.Module): The trained autoencoder model.
    adata (anndata.AnnData): The annotated data matrix containing gene names.
    top_n (int): The number of top contributing genes to return for both positive and negative contributions. Default is 10.

    Returns:
    dict: A dictionary where each key is a latent dimension, and the value is another dictionary with 'positive' and 'negative' keys containing lists of top contributing genes.
    """
    # Extract encoder weights
    encoder_weights = autoencoder.encoder.weight.detach().numpy()
    gene_names = adata.var_names

    top_genes = {}
    for i in range(encoder_weights.shape[0]):
        # Create a DataFrame for easier sorting
        df = pd.DataFrame(encoder_weights[i], index=gene_names, columns=['weight'])

        # Get top positive and negative contributing genes
        top_positive_genes = df[df['weight'] > 0].nlargest(top_n, 'weight').index.tolist()
        top_negative_genes = df[df['weight'] < 0].nsmallest(top_n, 'weight').index.tolist()

        top_genes[i] = {
            'positive': top_positive_genes,
            'negative': top_negative_genes
        }

    return top_genes