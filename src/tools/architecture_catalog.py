"""Catalog of buildable architecture patterns for AI+bio model design.

Each entry describes an architecture the agent can compose from, inspired
by AI+bio literature.  Entries include a PyTorch skeleton function that
returns an ``nn.Module`` given an architecture config dict.

The catalog is consumed by:
- ``build_architecture`` (model_builder.py) — looks up skeletons by name
- ``design_from_paper`` — matches paper-extracted concepts to catalog entries
- ``extract_architecture_from_paper`` (llm_tools.py) — grounds LLM output
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Reusable building blocks
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class ResidualBlock(nn.Module):
    """Feed-forward block with skip connection and layer norm."""

    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * hidden_mult, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class GraphAttentionLayer(nn.Module):
    """Single-head graph attention (Velickovic et al. 2018)."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: (N, in_features), adj: (N, N) binary or weighted adjacency."""
        h = self.W(x)
        N = h.size(0)
        h_i = h.unsqueeze(1).expand(N, N, -1)
        h_j = h.unsqueeze(0).expand(N, N, -1)
        e = self.leaky_relu(self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))
        mask = (adj == 0)
        e = e.masked_fill(mask, float("-inf"))
        alpha = torch.softmax(e, dim=-1)
        alpha = self.dropout(alpha)
        alpha = alpha.masked_fill(mask, 0.0)
        return alpha @ h


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between two modality embeddings."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(query, context, context)
        return self.norm(query + out)


# ---------------------------------------------------------------------------
# Skeleton builders — each returns an nn.Module
# ---------------------------------------------------------------------------

def build_gene_transformer(cfg: dict[str, Any]) -> nn.Module:
    """Self-attention over gene features, inspired by Enformer/scBERT.

    Handles both 3D sequence input (batch, seq_len, feat_dim) and 2D tabular
    input (batch, n_genes).  For 2D input each gene becomes a token with a
    scalar value projected to d_model — the biologically natural encoding where
    each gene is a position in the attention sequence.
    """
    input_dim = int(cfg.get("input_dim", 128))
    d_model = int(cfg.get("d_model", 128))
    n_heads = int(cfg.get("n_heads", 4))
    n_layers = int(cfg.get("n_layers", 3))
    n_classes = int(cfg.get("n_classes", 2))
    dropout = float(cfg.get("dropout", 0.1))
    max_len = int(cfg.get("max_len", 5000))

    class GeneTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.token_proj = nn.Linear(1, d_model)
            self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                # Tabular: (batch, n_genes) → each gene is a token
                h = self.token_proj(x.unsqueeze(-1))  # (batch, n_genes, d_model)
            else:
                h = self.input_proj(x)
            h = self.pos_enc(h)
            h = self.encoder(h)
            pooled = h.mean(dim=1)
            return self.classifier(pooled)

    return GeneTransformer()


def build_protein_interaction_gat(cfg: dict[str, Any]) -> nn.Module:
    """Multi-head GAT for protein interaction prediction."""
    in_features = int(cfg.get("input_dim", 64))
    hidden_dim = int(cfg.get("hidden_dim", 64))
    n_heads = int(cfg.get("n_heads", 4))
    n_classes = int(cfg.get("n_classes", 2))
    dropout = float(cfg.get("dropout", 0.1))

    class ProteinGAT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = nn.ModuleList([
                GraphAttentionLayer(in_features, hidden_dim, dropout) for _ in range(n_heads)
            ])
            self.out_proj = nn.Sequential(
                nn.Linear(hidden_dim * n_heads, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )

        def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            head_outs = [head(x, adj) for head in self.heads]
            h = torch.cat(head_outs, dim=-1)
            return self.out_proj(h)

    return ProteinGAT()


def build_multi_modal_encoder(cfg: dict[str, Any]) -> nn.Module:
    """Dual encoder with cross-attention for expression + text fusion."""
    expr_dim = int(cfg.get("expr_dim", 128))
    text_dim = int(cfg.get("text_dim", 768))
    d_model = int(cfg.get("d_model", 128))
    n_classes = int(cfg.get("n_classes", 2))
    n_heads = int(cfg.get("n_heads", 4))
    dropout = float(cfg.get("dropout", 0.1))

    class MultiModalEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.expr_proj = nn.Sequential(
                nn.Linear(expr_dim, d_model), nn.ReLU(), nn.LayerNorm(d_model),
            )
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, d_model), nn.ReLU(), nn.LayerNorm(d_model),
            )
            self.cross_attn = CrossAttentionFusion(d_model, n_heads, dropout)
            self.classifier = nn.Sequential(
                nn.Linear(d_model * 2, d_model), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(d_model, n_classes),
            )

        def forward(self, expr: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
            e = self.expr_proj(expr).unsqueeze(1)
            t = self.text_proj(text).unsqueeze(1)
            fused = self.cross_attn(e, t).squeeze(1)
            return self.classifier(torch.cat([fused, e.squeeze(1)], dim=-1))

    return MultiModalEncoder()


def build_residual_mlp(cfg: dict[str, Any]) -> nn.Module:
    """Deep residual MLP with skip connections."""
    input_dim = int(cfg.get("input_dim", 128))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    n_blocks = int(cfg.get("n_blocks", 4))
    n_classes = int(cfg.get("n_classes", 2))
    dropout = float(cfg.get("dropout", 0.1))

    class ResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.blocks = nn.Sequential(*[
                ResidualBlock(hidden_dim, hidden_mult=4, dropout=dropout)
                for _ in range(n_blocks)
            ])
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.blocks(self.input_proj(x)))

    return ResidualMLP()


def build_contrastive_encoder(cfg: dict[str, Any]) -> nn.Module:
    """SimCLR-style contrastive encoder for gene expression."""
    input_dim = int(cfg.get("input_dim", 128))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    proj_dim = int(cfg.get("proj_dim", 64))
    dropout = float(cfg.get("dropout", 0.1))

    class ContrastiveEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
            )
            self.projector = nn.Sequential(
                nn.Linear(hidden_dim, proj_dim), nn.ReLU(),
                nn.Linear(proj_dim, proj_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.encoder(x)
            return nn.functional.normalize(self.projector(h), dim=-1)

        def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

    return ContrastiveEncoder()


def build_vae_expression(cfg: dict[str, Any]) -> nn.Module:
    """Variational autoencoder for gene expression (scVI-inspired)."""
    input_dim = int(cfg.get("input_dim", 2000))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    latent_dim = int(cfg.get("latent_dim", 32))
    dropout = float(cfg.get("dropout", 0.1))

    class ExpressionVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            )
            self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
                nn.BatchNorm1d(hidden_dim // 2), nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return self.decoder(z)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

    return ExpressionVAE()


def build_gcn_message_passing(cfg: dict[str, Any]) -> nn.Module:
    """Manual GCN via sparse message-passing (no PyG dependency)."""
    input_dim = int(cfg.get("input_dim", 64))
    hidden_dim = int(cfg.get("hidden_dim", 64))
    n_layers = int(cfg.get("n_layers", 2))
    n_classes = int(cfg.get("n_classes", 2))
    dropout = float(cfg.get("dropout", 0.1))
    task = str(cfg.get("task", "node_classification"))

    class GCNLayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)

        def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
            return torch.relu(adj_norm @ self.linear(x))

    class ManualGCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dims = [input_dim] + [hidden_dim] * n_layers
            self.layers = nn.ModuleList([
                GCNLayer(dims[i], dims[i + 1]) for i in range(n_layers)
            ])
            self.dropout = nn.Dropout(dropout)
            if task == "graph_classification":
                self.head = nn.Linear(hidden_dim, n_classes)
            else:
                self.head = nn.Linear(hidden_dim, n_classes)

        @staticmethod
        def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
            """Symmetric normalization: D^{-1/2} A D^{-1/2}."""
            d = adj.sum(dim=1).clamp(min=1e-10)
            d_inv_sqrt = d.pow(-0.5)
            return (adj * d_inv_sqrt.unsqueeze(0)) * d_inv_sqrt.unsqueeze(1)

        def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            adj_norm = self.normalize_adj(adj + torch.eye(adj.size(0), device=adj.device))
            h = x
            for layer in self.layers:
                h = self.dropout(layer(h, adj_norm))
            if task == "graph_classification":
                h = h.mean(dim=0, keepdim=True)
            return self.head(h)

    return ManualGCN()


# ---------------------------------------------------------------------------
# Catalog dataclass and registry
# ---------------------------------------------------------------------------

@dataclass
class ArchitectureEntry:
    """A single architecture pattern in the catalog."""
    name: str
    paper: str
    components: list[str]
    input_type: str
    compatible_tasks: list[str]
    description: str
    skeleton_fn: Callable[[dict[str, Any]], nn.Module]
    default_config: dict[str, Any] = field(default_factory=dict)


ARCHITECTURE_CATALOG: list[ArchitectureEntry] = [
    ArchitectureEntry(
        name="gene_transformer",
        paper="Enformer (Avsec et al. 2021)",
        components=["positional_encoding", "multi_head_attention", "feed_forward", "layer_norm"],
        input_type="gene_expression_matrix",
        compatible_tasks=["gene_regulation_prediction", "classification", "expression_prediction"],
        description="Self-attention over gene features with positional encoding. Suitable for "
                    "capturing long-range dependencies in gene expression and regulatory sequences.",
        skeleton_fn=build_gene_transformer,
        default_config={"d_model": 128, "n_heads": 4, "n_layers": 3, "dropout": 0.1},
    ),
    ArchitectureEntry(
        name="protein_interaction_gat",
        paper="GAT (Velickovic et al. 2018)",
        components=["graph_attention", "multi_head_concat", "skip_connection", "readout"],
        input_type="ppi_graph + node_features",
        compatible_tasks=["interaction_prediction", "node_classification", "link_prediction"],
        description="Multi-head graph attention network for protein-protein interaction graphs. "
                    "Learns to attend over neighbours with different importance weights.",
        skeleton_fn=build_protein_interaction_gat,
        default_config={"hidden_dim": 64, "n_heads": 4, "dropout": 0.1},
    ),
    ArchitectureEntry(
        name="multi_modal_encoder",
        paper="CLIP-inspired dual encoder (Radford et al. 2021)",
        components=["dual_encoder", "cross_attention", "projection", "fusion"],
        input_type="expression_vector + text_embedding",
        compatible_tasks=["multi_modal_classification", "retrieval", "zero_shot"],
        description="Dual-encoder with cross-attention fusion for combining gene expression "
                    "features with text-derived embeddings (e.g. gene descriptions, paper abstracts).",
        skeleton_fn=build_multi_modal_encoder,
        default_config={"d_model": 128, "n_heads": 4, "expr_dim": 128, "text_dim": 768},
    ),
    ArchitectureEntry(
        name="residual_mlp",
        paper="ResNet-style MLP (He et al. 2016 adapted)",
        components=["linear", "skip_connection", "layer_norm", "gelu", "dropout"],
        input_type="feature_vector",
        compatible_tasks=["classification", "regression", "embedding"],
        description="Deep residual MLP with skip connections and layer normalization. "
                    "Strong baseline for tabular/expression data classification.",
        skeleton_fn=build_residual_mlp,
        default_config={"hidden_dim": 256, "n_blocks": 4, "dropout": 0.1},
    ),
    ArchitectureEntry(
        name="contrastive_encoder",
        paper="SimCLR (Chen et al. 2020)",
        components=["encoder", "projection_head", "normalize"],
        input_type="gene_expression_matrix",
        compatible_tasks=["embedding", "representation_learning", "clustering"],
        description="Contrastive learning encoder for gene expression. Learns representations "
                    "by maximizing agreement between augmented views of the same sample.",
        skeleton_fn=build_contrastive_encoder,
        default_config={"hidden_dim": 256, "proj_dim": 64, "dropout": 0.1},
    ),
    ArchitectureEntry(
        name="expression_vae",
        paper="scVI (Lopez et al. 2018)",
        components=["encoder", "reparameterize", "decoder", "batch_norm"],
        input_type="gene_expression_matrix",
        compatible_tasks=["reconstruction", "embedding", "generation", "imputation"],
        description="Variational autoencoder for single-cell or bulk gene expression data. "
                    "Inspired by scVI for learning latent biological representations.",
        skeleton_fn=build_vae_expression,
        default_config={"hidden_dim": 256, "latent_dim": 32, "dropout": 0.1},
    ),
    ArchitectureEntry(
        name="gcn_message_passing",
        paper="GCN (Kipf & Welling 2017)",
        components=["graph_convolution", "symmetric_norm", "relu", "dropout"],
        input_type="adjacency_matrix + node_features",
        compatible_tasks=["node_classification", "link_prediction", "graph_classification"],
        description="Manual GCN implementation using sparse message-passing. No PyG dependency. "
                    "Suitable for biological networks (PPI, gene regulatory, metabolic).",
        skeleton_fn=build_gcn_message_passing,
        default_config={"hidden_dim": 64, "n_layers": 2, "dropout": 0.1},
    ),
    ArchitectureEntry(
        name="attention_gene_network",
        paper="Graph Transformer (Yun et al. 2019)",
        components=["positional_encoding", "multi_head_attention", "graph_convolution", "feed_forward"],
        input_type="gene_expression_matrix",
        compatible_tasks=["gene_regulation_prediction", "classification", "network_inference"],
        description="Transformer-style self-attention applied to gene interaction networks. "
                    "Combines positional structure with attention-based feature aggregation.",
        skeleton_fn=build_gene_transformer,
        default_config={"d_model": 128, "n_heads": 4, "n_layers": 4, "dropout": 0.1},
    ),
    ArchitectureEntry(
        name="graphsage_bio",
        paper="GraphSAGE (Hamilton et al. 2017)",
        components=["neighborhood_sampling", "aggregation", "concat", "relu"],
        input_type="adjacency_matrix + node_features",
        compatible_tasks=["node_classification", "link_prediction", "inductive_learning"],
        description="Inductive GNN that samples and aggregates neighbour features. "
                    "Scales to large biological networks via neighbourhood sampling.",
        skeleton_fn=build_gcn_message_passing,
        default_config={"hidden_dim": 64, "n_layers": 2, "dropout": 0.1},
    ),
    ArchitectureEntry(
        name="denoising_autoencoder",
        paper="Denoising AE (Vincent et al. 2008)",
        components=["encoder", "noise_injection", "decoder", "reconstruction_loss"],
        input_type="gene_expression_matrix",
        compatible_tasks=["denoising", "imputation", "embedding", "feature_learning"],
        description="Denoising autoencoder that learns robust gene expression representations "
                    "by reconstructing clean inputs from corrupted versions.",
        skeleton_fn=build_vae_expression,
        default_config={"hidden_dim": 256, "latent_dim": 64, "dropout": 0.2},
    ),
    ArchitectureEntry(
        name="attention_pooling_classifier",
        paper="Attention pooling (Ilse et al. 2018)",
        components=["feature_extraction", "attention_pooling", "gated_attention", "classifier"],
        input_type="gene_expression_matrix",
        compatible_tasks=["classification", "multi_instance_learning"],
        description="Attention-based pooling classifier for gene expression. Uses gated "
                    "attention to weight gene contributions for sample-level prediction.",
        skeleton_fn=build_residual_mlp,
        default_config={"hidden_dim": 128, "n_blocks": 2, "dropout": 0.1},
    ),
    ArchitectureEntry(
        name="protein_lm_classifier",
        paper="ESM-2 (Lin et al. 2023)",
        components=["pretrained_encoder", "lora_adapter", "classification_head"],
        input_type="protein_sequence",
        compatible_tasks=["protein_function_prediction", "ppi_prediction", "property_regression"],
        description="ESM-2 protein language model with LoRA adapters for parameter-efficient "
                    "fine-tuning on downstream protein tasks.",
        skeleton_fn=build_residual_mlp,
        default_config={"hidden_dim": 320, "n_blocks": 1, "dropout": 0.1},
    ),
]

# Index by name for fast lookup
CATALOG_BY_NAME: dict[str, ArchitectureEntry] = {e.name: e for e in ARCHITECTURE_CATALOG}

# Component vocabulary for LLM grounding
COMPONENT_VOCABULARY: list[str] = sorted({
    comp for entry in ARCHITECTURE_CATALOG for comp in entry.components
})


def get_catalog_summary() -> str:
    """Return a compact text summary of the catalog for use in LLM prompts."""
    lines: list[str] = []
    for e in ARCHITECTURE_CATALOG:
        tasks_str = ", ".join(e.compatible_tasks)
        lines.append(f"- {e.name} ({e.paper}): {e.description[:100]}... Tasks: {tasks_str}")
    return "\n".join(lines)


def lookup_architecture(name: str) -> ArchitectureEntry | None:
    """Look up an architecture entry by name."""
    return CATALOG_BY_NAME.get(name)


def build_from_catalog(name: str, config: dict[str, Any]) -> nn.Module:
    """Build an nn.Module from catalog by name with config overrides."""
    entry = CATALOG_BY_NAME.get(name)
    if entry is None:
        raise ValueError(f"Unknown architecture: {name}. Available: {list(CATALOG_BY_NAME)}")
    merged = {**entry.default_config, **config}
    return entry.skeleton_fn(merged)


def match_paper_concepts(
    components: list[str],
    task: str | None = None,
) -> list[ArchitectureEntry]:
    """Find catalog entries matching extracted paper components and task."""
    components_lower = {c.lower() for c in components}
    scored: list[tuple[float, ArchitectureEntry]] = []
    for entry in ARCHITECTURE_CATALOG:
        entry_comps = {c.lower() for c in entry.components}
        overlap = len(components_lower & entry_comps)
        if overlap == 0:
            continue
        score = overlap / max(len(components_lower), 1)
        if task and task.lower() in [t.lower() for t in entry.compatible_tasks]:
            score += 0.3
        scored.append((score, entry))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [e for _, e in scored]
