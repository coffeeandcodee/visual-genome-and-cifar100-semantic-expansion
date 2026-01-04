"""
Lab 6: Skip-Gram with Negative Sampling (SGNS) for Network Embeddings

Implements Skip-Gram with Negative Sampling to learn embeddings from text networks.
Includes training, evaluation, and visualization tools.

KEY FEATURES:
1. Filters punctuation tokens to prevent hub poisoning
2. Proper negative sampling (5-20 negatives per positive)
3. Weighted sampling by co-occurrence frequency
4. Anti-overfitting: dropout, weight decay, label smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm       #Comes from arabic word taqaddum interestingly enough
from sklearn.manifold import TSNE
import networkx as nx
import requests
import zipfile
import json
import os
from typing import List, Dict, Set, Tuple

import unittest
from collections import Counter


# ============================================================================
# Utilities
# ============================================================================

def download_file(url, out_path):
    """Download a file from URL."""
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded to {out_path}")


def prepare_visual_genome_text(zip_url, zip_path="region_descriptions.json.zip", 
                                json_path="region_descriptions.json",
                                output_path="vg_text.txt"):
    """Download, unzip, and process Visual Genome region descriptions."""
    
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping processing.")
        return output_path

    if not os.path.exists(zip_path):
        download_file(zip_url, zip_path)
    
    if not os.path.exists(json_path):
        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
    
    print(f"Processing {json_path} into {output_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    phrases = [region['phrase'] for img in data for region in img['regions']]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(" . ".join(phrases))
    
    print(f"Processed {len(phrases):,} phrases into {output_path}")
    return output_path


def filter_punctuation_from_network(network_data, punctuation_tokens={'.', ',', '<RARE>', "'"}):
    """
    Remove punctuation tokens from network to prevent hub poisoning.
    
    Punctuation creates massive hubs that bridge unrelated sentences,
    poisoning the graph structure and making embeddings meaningless.
    """
    original_graph = network_data['graph']
    original_nodes = network_data['nodes']
    original_distance_matrix = network_data['distance_matrix']
    
    # Filter nodes
    filtered_nodes = [n for n in original_nodes if n not in punctuation_tokens]
    old_indices = [i for i, n in enumerate(original_nodes) if n not in punctuation_tokens]
    
    # Filter matrices
    filtered_distance_matrix = original_distance_matrix[np.ix_(old_indices, old_indices)]
    
    # Create filtered graph
    filtered_graph = nx.Graph()
    filtered_graph.add_nodes_from(filtered_nodes)
    for u, v in original_graph.edges():
        if u in filtered_nodes and v in filtered_nodes:
            filtered_graph.add_edge(u, v)
    
    print(f"\n🔧 PUNCTUATION FILTER:")
    print(f"  Removed: {punctuation_tokens}")
    print(f"  Nodes: {len(original_nodes):,} → {len(filtered_nodes):,}")
    print(f"  Edges: {original_graph.number_of_edges():,} → {filtered_graph.number_of_edges():,}")
    
    return {
        **network_data,
        'graph': filtered_graph,
        'nodes': filtered_nodes,
        'distance_matrix': filtered_distance_matrix
    }


# ============================================================================
# Dataset
# ============================================================================


class SkipGramDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        graph: nx.Graph,
        nodes: List[str],
        distance_matrix: np.ndarray,
        num_negative: int = 15,
        context_size: int = 1,
    ):
    
        super().__init__()
        
        self.graph = graph
        self.nodes = nodes
        self.node_to_idx = {node: i for i, node in enumerate(nodes)}
        self.vocab_size = len(nodes)
        self.num_negative = num_negative
        self.distance_matrix = distance_matrix

        # Step 1: Build context sets for each node. WHY: We need to know which nodes are "related" to create positive pairs
        self.contexts = self._build_contexts(context_size)

        # Step 2: Convert contexts into training pairs and compute weights.  PyTorch needs explicit (center, context) pairs, and weighting helps
        #  the model focus on more important relationships
        self.pairs, self.weights = self._generate_weighted_pairs()

        
        # Step 3: Initialize per-worker RNG (lazily, in __getitem__)
        # WHY: Multi-worker DataLoaders need independent random streams
        self._local_rng = None
        
        self._print_stats()

    # ========================================================================
    # TODO: IMPLEMENT THESE METHODS
    # ========================================================================

    def _build_contexts(self, context_size: int) -> Dict[str, Set[str]]:
 
        #There might be a better way of doing this
        contexts = {}  #*

        for node in self.nodes:  #*
            if node not in self.graph:  #*
                contexts[node] = set()  #*
                continue  #*
            
            distances = nx.single_source_shortest_path_length(self.graph, node, cutoff=context_size)  #*
            contexts[node] = {n for n, d in distances.items() if d > 0 and n in self.node_to_idx}  #*
            
        return contexts  #*

    def _generate_weighted_pairs(self) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Generate (center_idx, context_idx) pairs and compute importance weights.
        
        Algorithm:
        1. Iterate through self.contexts to create pairs
        2. For each (center, context) pair, look up the distance from distance_matrix
        3. Convert distances to weights (closer pairs = higher weight)
        4. Apply transformations to prevent overfitting:
           - Sublinear scaling (sqrt) to reduce extreme weights
           - Clipping to prevent dominance by a few pairs
        5. Normalize weights for interpretability
        
        WEIGHT FORMULA:
        Starting from raw distances (where larger = farther):
            raw_weight = (max_distance + 1) - distance  # invert so closer = larger
            weight = sqrt(raw_weight)                    # sublinear scaling
            weight = clip(weight, max=95th_percentile * 3)  # prevent outliers
            weight = normalize(weight)                   # scale to reasonable range
        
        WHY THESE TRANSFORMS:
        - Inversion: Skip-gram should focus on close relationships
        - Sqrt: Prevents a few very-high-frequency pairs from dominating
        - Clipping: Extreme weights can cause training instability
        - Normalization: Makes weights interpretable when printed
        
        HINTS:
        - If there are no pairs, return ([], np.array([], dtype=np.float32))
        - Use self.node_to_idx to convert node strings to indices
        - np.percentile(weights, 95) gives you the 95th percentile
        - Always ensure weights >= 1e-6 to avoid zeros
        
        Returns:
            pairs: List of (center_idx, context_idx) tuples
            weights: numpy array of weights, same length as pairs
        """
        pairs = []  #*
        raw_distances = []  #*
        
        for center_word, context_words in self.contexts.items():  #*
            center_idx = self.node_to_idx[center_word]  #*
            for context_word in context_words:  #*
                context_idx = self.node_to_idx[context_word]  #*
                pairs.append((center_idx, context_idx))  #*
                raw_distances.append(self.distance_matrix[center_idx, context_idx])  #*
        
        if len(pairs) == 0:  #*
            return [], np.array([], dtype=np.float32)  #*
        
        raw_distances = np.array(raw_distances, dtype=np.float32)  #*
        weights = (raw_distances.max() + 1) - raw_distances  #*
        weights = np.sqrt(weights)  #*
        weights = np.clip(weights, None, np.percentile(weights, 95) * 3)  #*
        weights = weights / weights.sum() * len(weights)  #*
        weights = np.maximum(weights, 1e-6)  #*
        
        return pairs, weights.astype(np.float32)  #*

    def __getitem__(self, idx: int) -> Tuple[np.int64, np.int64, np.ndarray]:
        """
        Get a single training example: (center_idx, context_idx, negatives).
        
        Algorithm:
        1. Initialize per-worker RNG if needed (for DataLoader multi-processing)
        2. Retrieve the positive pair at index idx
        3. Build exclusion set: center + all its true contexts
        4. Sample num_negative nodes from vocabulary, excluding the exclusion set
        5. Return (center_idx, context_idx, negatives_array)
        
        WHY EXCLUDE TRUE CONTEXTS:
        If we use a true context node as a "negative" example, we're training
        the model with contradictory signals (it's both positive and negative).
        This confuses learning.
        
        WHY PER-WORKER RNG:
        DataLoader uses multiple worker processes. Each needs an independent
        random stream or they'll all generate identical "random" samples.
        
        HINTS:
        - Use torch.utils.data.get_worker_info() to detect multi-worker mode
        - Seed the RNG with: torch.initial_seed() + worker_id
        - Use self._local_rng.choice() for sampling negatives
        - If available pool < num_negative, use replace=True
        - Handle edge case: if no nodes are available, use the whole vocab
        
        Args:
            idx: Index into self.pairs
            
        Returns:
            center_idx: numpy int64 scalar
            context_idx: numpy int64 scalar  
            negatives: numpy int64 array of shape (num_negative,)
        """
        if self._local_rng is None:  #*
            worker_info = torch.utils.data.get_worker_info()  #*
            seed = (torch.initial_seed() + (worker_info.id if worker_info else 0)) % (2**32)  #*
            self._local_rng = np.random.RandomState(seed)  #*
        
        center_idx, context_idx = self.pairs[idx]  #*
        center_node = self.nodes[center_idx]  #*
        excluded = {self.node_to_idx[n] for n in self.contexts[center_node]}  #*
        excluded.add(center_idx)  #*
        available = np.array([i for i in range(self.vocab_size) if i not in excluded], dtype=np.int64)  #*
        
        if len(available) == 0:  #*
            available = np.arange(self.vocab_size, dtype=np.int64)  #*
        
        replace = len(available) < self.num_negative  #*
        negatives = self._local_rng.choice(available, size=self.num_negative, replace=replace)  #*
        
        return np.int64(center_idx), np.int64(context_idx), negatives.astype(np.int64)  #*

    # ========================================================================
    # PROVIDED HELPER METHODS (no changes needed)
    # ========================================================================

    def get_sample_weights(self) -> np.ndarray:
        """
        Return per-pair weights for WeightedRandomSampler.
        
        Usage:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=dataset.get_sample_weights(),
                num_samples=len(dataset),
                replacement=True
            )
            loader = DataLoader(dataset, sampler=sampler, batch_size=32)
        """
        return self.weights

    def __len__(self) -> int:
        """Number of positive training pairs."""
        return len(self.pairs)

    def _print_stats(self):
        """Print dataset statistics for debugging."""
        print("\n📊 SkipGramDataset Statistics:")
        print(f"  Vocabulary size: {self.vocab_size:,}")
        print(f"  Positive pairs: {len(self.pairs):,}")
        print(f"  Negatives per positive: {self.num_negative}")
        print(f"  Total samples per epoch: {len(self.pairs) * (1 + self.num_negative):,}")
        
        if self.weights.size > 0:
            print(f"\n  Weight distribution:")
            print(f"    Min: {self.weights.min():.6f}")
            print(f"    Mean: {self.weights.mean():.6f}")
            print(f"    Median: {np.median(self.weights):.6f}")
            print(f"    Max: {self.weights.max():.6f}")
        else:
            print("  ⚠️  No pairs found - check your graph and nodes!")


# ============================================================================
# Model
# ============================================================================

"""
SkipGramModel - Student Starter Code
=====================================

LEARNING OBJECTIVES:
1. Understand dual embedding spaces (center vs context) in Skip-Gram
2. Implement negative sampling loss with label smoothing
3. Learn proper weight initialization for embedding layers
4. Master PyTorch's batched matrix operations

WHAT YOU'LL IMPLEMENT:
- [ ] _init_embeddings(): Initialize embedding weights properly
- [ ] forward(): Compute Skip-Gram Negative Sampling (SGNS) loss
- [ ] get_embeddings(): Extract learned embeddings for downstream use

KEY CONCEPTS:
- Center embeddings: Represent words as "query" vectors
- Context embeddings: Represent words as "key" vectors  
- Why two spaces? Asymmetry helps distinguish "is context of" from "has context"
- Negative sampling: Contrastive learning - push apart unrelated pairs
"""

class SkipGramModel(nn.Module):
    """
    Skip-Gram model with Negative Sampling (SGNS).
    
    Architecture:
        - center_embeddings: Embedding(V, D) - represents words as query vectors
        - context_embeddings: Embedding(V, D) - represents words as key vectors
        - dropout: Regularization applied to center embeddings
    
    Why two embedding matrices?
        In Skip-Gram, words play two roles:
        1. As CENTER: "What contexts does this word appear in?"
        2. As CONTEXT: "What centers is this word a context for?"
        
        These are asymmetric relationships. Using separate embeddings lets the
        model learn different representations for each role, improving quality.
    
    Training objective:
        Maximize: P(context | center) for true pairs
        Minimize: P(negative | center) for random pairs
        
    Example:
        >>> model = SkipGramModel(vocab_size=1000, embedding_dim=128)
        >>> center = torch.tensor([5, 10])      # batch of 2 center words
        >>> context = torch.tensor([8, 15])     # their true contexts
        >>> negatives = torch.randint(0, 1000, (2, 10))  # 10 negatives each
        >>> loss = model(center, context, negatives)
        >>> print(loss.shape)  # torch.Size([2]) - loss per example
    """

    def __init__(self, vocab_size: int, embedding_dim: int, dropout: float = 0.3):
        """
        Initialize Skip-Gram model.
        
        Args:
            vocab_size: Size of vocabulary (number of unique nodes/words)
            embedding_dim: Dimensionality of embedding vectors (typically 50-300)
            dropout: Dropout probability for regularization (prevents overfitting)
        """
        super().__init__()
        
        # Two embedding matrices: one for center words, one for context words
        # WHY: Asymmetric roles in Skip-Gram (see class docstring)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Dropout for regularization (applied only to center embeddings during training)
        # WHY: Prevents model from memorizing training pairs, improves generalization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings with proper scaling
        self._init_embeddings()

    def _init_embeddings(self):
        """
        Initialize embedding weights using uniform distribution.
        
        Why initialization matters:
            - Too large: Training becomes unstable (exploding gradients)
            - Too small: Learning is slow (vanishing gradients)  
            - Rule of thumb: scale inversely with embedding dimension
        
        Standard practice for Skip-Gram:
            - Use uniform distribution: U(-scale, scale)
            - Scale = 0.5 / sqrt(embedding_dim) OR 0.5 / embedding_dim
            - We use 0.5 / embedding_dim for slightly more conservative init
        
        TODO: Initialize both embedding matrices
        HINTS:
        - Access embedding dimension via: self.center_embeddings.embedding_dim
        - Use nn.init.uniform_(tensor, low, high) to initialize in-place
        - Apply same initialization to both center_embeddings and context_embeddings
        """
        scale = 0.5 / self.center_embeddings.embedding_dim  #*
        nn.init.uniform_(self.center_embeddings.weight, -scale, scale)  #*
        nn.init.uniform_(self.context_embeddings.weight, -scale, scale)  #*

    #  NOTE  forwad function implicitly called at loss = model(centers, contexts, negs, True, label_smoothing).mean()
    def forward(
        self, 
        center: torch.Tensor,      # shape: (batch_size,)
        context: torch.Tensor,     # shape: (batch_size,)
        negatives: torch.Tensor,   # shape: (batch_size, num_negatives)
        apply_dropout: bool = True,
        label_smoothing: float = 0.1
    ) -> torch.Tensor:
        """
        Compute Skip-Gram Negative Sampling loss.
        
        Algorithm:
        1. Look up embeddings for center, context, and negative words
        2. Compute positive score: similarity(center, context)
        3. Compute negative scores: similarity(center, each negative)
        4. Apply label smoothing to targets (anti-overfitting)
        5. Compute binary cross-entropy loss using log-sigmoid
        6. Return negative loss (we'll minimize this, which maximizes log-likelihood)
        
        Mathematical formulation:
            Positive loss: -log(σ(center · context))
            Negative loss: -Σ log(σ(-center · negative_i))
            
            With label smoothing (α = 0.1):
            - True positive target: 0.9 instead of 1.0
            - True negative target: 0.9 instead of 1.0
            This prevents overconfident predictions
        
        Args:
            center: Batch of center word indices, shape (B,)
            context: Batch of true context word indices, shape (B,)
            negatives: Batch of negative word indices, shape (B, K)
            apply_dropout: Whether to apply dropout to center embeddings
            label_smoothing: Smoothing factor (0 = no smoothing, 0.1 = mild)
            
        Returns:
            loss: Per-example loss, shape (B,). Caller typically does loss.mean()
        
        HINTS:
        - Use self.center_embeddings(center) to look up embeddings
        - Dot product: torch.sum(a * b, dim=1) for element-wise mult + sum
        - Batch matrix multiply: torch.bmm(A, B) where A is (B,K,D), B is (B,D,1)
        - Log-sigmoid: F.logsigmoid(x) computes log(1/(1+exp(-x))) stably
        - Label smoothing formula: smoothed_target = (1 - α) for positive
        """
        
        center_emb = self.dropout(self.center_embeddings(center)) if apply_dropout else self.center_embeddings(center)  #*
        context_emb = self.context_embeddings(context)  #*
        negative_emb = self.context_embeddings(negatives)  #*
        
        pos_score = torch.sum(center_emb * context_emb, dim=1)  #*
        neg_score = torch.bmm(negative_emb, center_emb.unsqueeze(2)).squeeze(2)  #*
        
        pos_target = 1.0 - label_smoothing  #*
        pos_loss = -(pos_target * F.logsigmoid(pos_score) + (1.0 - pos_target) * F.logsigmoid(-pos_score))  #*
        
        neg_target = 1.0 - label_smoothing  #*
        neg_loss = -(neg_target * F.logsigmoid(-neg_score) + (1.0 - neg_target) * F.logsigmoid(neg_score)).sum(dim=1)  #*
        
        # Return the positive loss to minimize (don't negate again!)
        return (pos_loss + neg_loss)  #*

    def get_embeddings(self) -> np.ndarray:
        """
        Extract the learned center embeddings as a numpy array.
        
        Why center embeddings?
            Both center and context embeddings contain learned information, but:
            - Center embeddings are what we optimized as "query" vectors
            - They're used during training with dropout (more robust)
            - Convention: use center embeddings for downstream tasks
            
        Alternative: You could average center + context embeddings, but this
        is less common and may not improve quality.
        
        Returns:
            embeddings: numpy array of shape (vocab_size, embedding_dim)
        
        TODO: Extract center embeddings and convert to numpy
        HINTS:
        - Use .weight to access the embedding matrix
        - Use .detach() to remove from computation graph
        - Use .cpu() to move to CPU (in case model is on GPU)
        - Use .numpy() to convert to numpy array
        """
        return self.center_embeddings.weight.detach().cpu().numpy()  #*



# ============================================================================
# Training
# ============================================================================

def train_embeddings(
    network_data,
    embedding_dim=128,
    batch_size=512,
    epochs=20,
    learning_rate=0.001,
    num_negative=15,
    validation_fraction=0.05,
    context_size=1,
    dropout=0.3,
    weight_decay=1e-4,
    label_smoothing=0.1,
    patience=3,
    device=None,
    save_plot=True
):
    """
    Train Skip-Gram embeddings with weighted sampling.
    
    Args:
        network_data: Dict with 'graph', 'nodes', 'distance_matrix'
        embedding_dim: Embedding dimensionality
        batch_size: Training batch size
        epochs: Maximum epochs
        learning_rate: Initial learning rate
        num_negative: Negatives per positive (5-20 recommended)
        validation_fraction: Fraction for validation
        context_size: Graph distance for context (1=neighbors)
        dropout: Dropout rate (default: 0.3)
        weight_decay: L2 regularization (default: 1e-4)
        label_smoothing: Label smoothing factor (default: 0.1)
        patience: Early stopping patience
        device: 'cuda' or 'cpu'
        save_plot: Save training curve
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu") #How do I set .is_available() to true?
    
    # Filter punctuation
    network_data = filter_punctuation_from_network(network_data)
    nodes = network_data['nodes']
    graph = network_data['graph']
    distance_matrix = network_data['distance_matrix']
    
    # Split edges
    all_edges = list(graph.edges())
    np.random.shuffle(all_edges)
    split_idx = int(len(all_edges) * (1 - validation_fraction))
    
    train_graph = nx.Graph()
    train_graph.add_nodes_from(nodes)
    train_graph.add_edges_from(all_edges[:split_idx])
    
    val_graph = nx.Graph()
    val_graph.add_nodes_from(nodes)
    val_graph.add_edges_from(all_edges[split_idx:])
    
    print(f"\nTrain edges: {len(all_edges[:split_idx]):,}, Val edges: {len(all_edges[split_idx:]):,}")
    
    # Create datasets
    train_dataset = SkipGramDataset(train_graph, nodes, distance_matrix, num_negative, context_size)
    val_dataset = SkipGramDataset(val_graph, nodes, distance_matrix, num_negative, context_size)
    
    # Create loaders with weighted sampling
    sampler = WeightedRandomSampler(
        weights=train_dataset.get_sample_weights(),
        num_samples=len(train_dataset),
        replacement=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # Initialize model
    model = SkipGramModel(len(nodes), embedding_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    print(f"\nTraining on {device}")
    print(f"Vocab: {len(nodes)}, Embed dim: {embedding_dim}, Context: {context_size}, Negatives: {num_negative}")
    print(f"Regularization: dropout={dropout}, weight_decay={weight_decay}, label_smoothing={label_smoothing}")
    
    # Training loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}", leave=False)
        for i, (centers, contexts, negs) in train_pbar:
            centers, contexts, negs = centers.to(device), contexts.to(device), negs.to(device)
            
            loss = model(centers, contexts, negs, True, label_smoothing).mean()
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({'train_loss': f'{total_loss / (i + 1):.4f}'})
        
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        total_val_loss = 0.0
        
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating", leave=False)
        with torch.no_grad():
            for i, (centers, contexts, negs) in val_pbar:
                centers, contexts, negs = centers.to(device), contexts.to(device), negs.to(device)
                
                batch_loss = model(centers, contexts, negs, False, 0.0).mean().item()
                total_val_loss += batch_loss
                val_pbar.set_postfix({'val_loss': f'{total_val_loss / (i + 1):.4f}'})
        
        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch:02d}  train={train_loss:.4f}  val={val_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  
            best_model_state = model.state_dict()          
            save_data = {
                'model_state_dict': best_model_state,
                'nodes': nodes,
                'vocab_size': len(nodes),
                'embedding_dim': embedding_dim
            }
            torch.save(save_data, "best_model523.pth")        
            print(f"  → Best model (val_loss={best_val_loss:.4f}), saved to best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save plot
    if save_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, 'o-', label='Train', linewidth=2, markersize=6)
        plt.plot(val_losses, 's-', label='Validation', linewidth=2, markersize=6)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_loss.png', dpi=150)
        print("\nSaved loss plot to training_loss.png")
        plt.close()
    
    return {
        'nodes': nodes,
        'embeddings': model.get_embeddings(),
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses
    }


import time # Needed for timestamping

def train_embeddings2(
    network_data,
    embedding_dim=128,
    batch_size=512,
    epochs=20,
    learning_rate=0.001,
    num_negative=15,
    validation_fraction=0.05,
    context_size=1,
    dropout=0.3,
    weight_decay=1e-4,
    label_smoothing=0.1,
    patience=3,
    device=None,
    save_plot=True
):
    # 1. Capture Hyperparameters immediately for the log
    # locals() grabs all arguments passed to the function
    hyperparams = locals().copy()
    if 'network_data' in hyperparams: del hyperparams['network_data'] # Don't print the whole graph!

    # ... [YOUR EXISTING SETUP CODE] ...
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # [Start Timer]
    start_time = time.time()
    
    # ... [ALL YOUR EXISTING LOADING/MODEL CODE] ...
    # (I'm skipping the middle parts to keep this response short, 
    #  but imagine your exact code is here)
    
    # ... [AFTER THE TRAINING LOOP FINISHES] ...
    
    total_time = time.time() - start_time
    
    # --- NEW: SAVE LOG FILE ---
    log_filename = "training_summary.txt"
    
    with open(log_filename, "w") as f:
        f.write("="*40 + "\n")
        f.write("      TRAINING EXPERIMENT REPORT\n")
        f.write("="*40 + "\n\n")
        
        f.write("--- HYPERPARAMETERS ---\n")
        for key, value in hyperparams.items():
            f.write(f"{key:<20}: {value}\n")
            
        f.write("\n--- RESULTS ---\n")
        f.write(f"Best Val Loss       : {best_val_loss:.4f}\n")
        f.write(f"Final Train Loss    : {train_losses[-1]:.4f}\n")
        f.write(f"Total Epochs Run    : {epoch}\n")
        f.write(f"Training Time       : {total_time/60:.2f} minutes\n")
        f.write(f"Vocabulary Size     : {len(nodes)}\n")
        f.write(f"Device Used         : {device}\n")
        
        f.write("\n--- LOSS HISTORY ---\n")
        f.write(f"Train Losses: {[round(x, 4) for x in train_losses]}\n")
        f.write(f"Val Losses:   {[round(x, 4) for x in val_losses]}\n")

    print(f"\n📝 Experiment log saved to {log_filename}")

    return {
        'nodes': nodes,
        'embeddings': model.get_embeddings(),
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses
    }



# ============================================================================
# Analysis
# ============================================================================

def find_similar_words(word, nodes, embeddings, top_k=10):
    """Find most similar words using cosine similarity."""
    if word not in nodes:
        return []
    
    idx = nodes.index(word)
    target_vec = embeddings[idx]
    
    similarities = (embeddings @ target_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(target_vec) + 1e-10)
    top_indices = np.argsort(-similarities)[1:top_k+1]
    
    return [(nodes[i], float(similarities[i])) for i in top_indices]


def solve_analogy(word_a, word_b, word_c, nodes, embeddings, top_k=5):
    """Solve word analogies: word_a is to word_b as word_c is to ?"""
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    if not all(w in node_to_idx for w in [word_a, word_b, word_c]):
        return []
    
    target_vec = embeddings[node_to_idx[word_b]] - embeddings[node_to_idx[word_a]] + embeddings[node_to_idx[word_c]]
    similarities = (embeddings @ target_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(target_vec) + 1e-10)
    
    exclude = {node_to_idx[w] for w in [word_a, word_b, word_c]}
    results = [(nodes[i], float(similarities[i])) for i in np.argsort(-similarities) if i not in exclude][:top_k]
    
    return results


def visualize_embeddings(nodes, embeddings, output_file="embeddings_tsne.png", 
                        sample_size=200, annotate=True):
    """Create t-SNE visualization of embeddings."""
    n_samples = min(sample_size, len(nodes))
    selected_embeddings = embeddings[:n_samples]
    selected_nodes = nodes[:n_samples]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
    projection = tsne.fit_transform(selected_embeddings)
    
    plt.figure(figsize=(14, 14))
    plt.scatter(projection[:, 0], projection[:, 1], s=40, alpha=0.6, c='steelblue')
    
    if annotate:
        for i, word in enumerate(selected_nodes):
            plt.annotate(word, (projection[i, 0], projection[i, 1]), fontsize=9, alpha=0.8)
    
    plt.title(f"t-SNE Visualization of Top {n_samples} Word Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved t-SNE to {output_file}")
    plt.close()


def analyze_embeddings(nodes, embeddings, 
                       similarity_examples=None,
                       analogy_examples=None,
                       cluster_seeds=None):
    """Comprehensive analysis of learned embeddings."""
    print("\n" + "="*80)
    print("EMBEDDING ANALYSIS")
    print("="*80)
    
    print(f"\nVocabulary: {len(nodes):,}  Embedding dim: {embeddings.shape[1]}")
    
    # Similarity statistics
    sample_emb = embeddings[:min(100, len(embeddings))]
    norms = np.linalg.norm(sample_emb, axis=1, keepdims=True)
    normalized = sample_emb / (norms + 1e-10)
    sim_matrix = normalized @ normalized.T
    sim_values = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    
    print(f"\nSimilarity stats (100 word sample):")
    print(f"  Mean: {sim_values.mean():.4f}  Std: {sim_values.std():.4f}")
    print(f"  Min: {sim_values.min():.4f}  Max: {sim_values.max():.4f}")
    
    # Nearest neighbors
    if similarity_examples:
        print("\n" + "="*80)
        print("NEAREST NEIGHBORS")
        print("="*80)
        for word in similarity_examples:
            similar = find_similar_words(word, nodes, embeddings, top_k=8)
            print(f"\nMost similar to '{word}':")
            if not similar:
                print("  (not in vocabulary)")
            else:
                for token, score in similar:
                    print(f"  {token:15s}  similarity={score:.4f}")
    
    # Analogies
    if analogy_examples:
        print("\n" + "="*80)
        print("WORD ANALOGIES (a:b :: c:?)")
        print("="*80)
        for a, b, c in analogy_examples:
            results = solve_analogy(a, b, c, nodes, embeddings, top_k=3)
            print(f"\n{a}:{b} :: {c}:?")
            if results:
                for token, score in results:
                    print(f"  {token:15s}  score={score:.4f}")
            else:
                print("  (words not in vocabulary)")
    
    # Semantic clusters
    if cluster_seeds:
        print("\n" + "="*80)
        print("SEMANTIC CLUSTERS")
        print("="*80)
        for seed in cluster_seeds:
            if seed in nodes:
                cluster = find_similar_words(seed, nodes, embeddings, top_k=5)
                print(f"\n'{seed}': {', '.join([w for w, _ in cluster])}")
    
    print("\n" + "="*80)


"""
Unit Tests for Skip-Gram with Negative Sampling

Starter skeleton for testing SkipGramDataset and SkipGramModel classes.
Students should implement their own test cases inside the provided class structures.
"""

class TestSkipGramDataset(unittest.TestCase):
    """Tests for SkipGramDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test graph
        self.graph = nx.Graph()
        self.graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")])
        self.nodes = ["a", "b", "c", "d"]
        self.distance_matrix = np.ones((4, 4), dtype=np.float32)
        np.fill_diagonal(self.distance_matrix, 0)
    
    def test_contexts_built(self):  #*
        """Test that contexts are built correctly."""  #*
        dataset = SkipGramDataset(self.graph, self.nodes, self.distance_matrix, num_negative=2, context_size=1)  #*
        self.assertIsInstance(dataset.contexts, dict)  #*
        self.assertEqual(len(dataset.contexts), len(self.nodes))  #*
        self.assertGreater(len(dataset.contexts["a"]), 0)  #*
    
    def test_pairs_generated(self):  #*
        """Test that pairs are generated."""  #*
        dataset = SkipGramDataset(self.graph, self.nodes, self.distance_matrix, num_negative=2, context_size=1)  #*
        self.assertGreater(len(dataset.pairs), 0)  #*
        self.assertEqual(len(dataset.pairs), len(dataset.weights))  #*
    
    def test_getitem_returns_triplet(self):  #*
        """Test that __getitem__ returns correct format."""  #*
        dataset = SkipGramDataset(self.graph, self.nodes, self.distance_matrix, num_negative=3, context_size=1)  #*
        center, context, negatives = dataset[0]  #*
        self.assertIsInstance(center, (int, np.integer))  #*
        self.assertIsInstance(context, (int, np.integer))  #*
        self.assertEqual(len(negatives), 3)  #*


class TestSkipGramModel(unittest.TestCase):
    """Tests for SkipGramModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 100
        self.embedding_dim = 32
        self.model = SkipGramModel(self.vocab_size, self.embedding_dim, dropout=0.3)
    
    def test_embeddings_initialized(self):  #*
        """Test that embeddings are initialized."""  #*
        center_weights = self.model.center_embeddings.weight  #*
        context_weights = self.model.context_embeddings.weight  #*
        self.assertEqual(center_weights.shape, (self.vocab_size, self.embedding_dim))  #*
        self.assertEqual(context_weights.shape, (self.vocab_size, self.embedding_dim))  #*
        self.assertNotEqual(center_weights.abs().sum().item(), 0)  #*
    
    def test_forward_returns_loss(self):  #*
        """Test that forward pass returns loss."""  #*
        batch_size = 16  #*
        num_neg = 5  #*
        centers = torch.randint(0, self.vocab_size, (batch_size,))  #*
        contexts = torch.randint(0, self.vocab_size, (batch_size,))  #*
        negatives = torch.randint(0, self.vocab_size, (batch_size, num_neg))  #*
        loss = self.model(centers, contexts, negatives)  #*
        self.assertEqual(loss.shape, (batch_size,))  #*
    
    def test_get_embeddings_returns_numpy(self):  #*
        """Test that get_embeddings returns numpy array."""  #*
        embeddings = self.model.get_embeddings()  #*
        self.assertIsInstance(embeddings, np.ndarray)  #*
        self.assertEqual(embeddings.shape, (self.vocab_size, self.embedding_dim))  #*    


class TestIntegration(unittest.TestCase):
    """Integration tests for dataset and model working together."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.graph = nx.karate_club_graph()
        self.nodes = list(self.graph.nodes())
        self.nodes = [str(n) for n in self.nodes]
        n = len(self.nodes)
        self.distance_matrix = np.random.rand(n, n).astype(np.float32)
        np.fill_diagonal(self.distance_matrix, 0)
    
    def test_dataset_model_compatibility(self):  #*
        """Test that dataset output works with model input."""  #*
        dataset = SkipGramDataset(self.graph, self.nodes, self.distance_matrix, num_negative=5, context_size=1)  #*
        model = SkipGramModel(len(self.nodes), embedding_dim=16)  #*
        
        if len(dataset) > 0:  #*
            center, context, negatives = dataset[0]  #*
            centers = torch.tensor([center])  #*
            contexts = torch.tensor([context])  #*
            negs = torch.tensor([negatives])  #*
            loss = model(centers, contexts, negs)  #*
            self.assertEqual(loss.shape, (1,))  #*
            self.assertTrue(torch.isfinite(loss).all())  #*    


def run_tests():
    """Run all unit tests."""
    print("=" * 70)
    print("RUNNING SKIP-GRAM UNIT TESTS")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestSkipGramDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestSkipGramModel))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        print(f"Total tests run: {result.testsRun}")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
