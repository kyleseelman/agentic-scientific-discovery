from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.tools.data_analysis import ToolContext


def plot_volcano(ctx: ToolContext, params: dict[str, Any]) -> str:
    fdr = float(params.get("fdr_threshold", 0.05))
    de_path = ctx.output_dir / "differential_expression.csv"
    if not de_path.exists():
        raise FileNotFoundError("differential_expression.csv required for volcano plot")
    de = pd.read_csv(de_path, index_col=0)
    x = de["log2_fold_change"].values
    y = -np.log10(de["qvalue"].clip(lower=1e-300))
    sig = de["qvalue"].values < fdr
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x[~sig], y[~sig], s=10, c="#94a3b8", alpha=0.6, label="not sig")
    ax.scatter(x[sig], y[sig], s=14, c="#dc2626", alpha=0.8, label=f"q < {fdr}")
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10 q")
    ax.set_title("Volcano plot")
    ax.legend(frameon=False)
    out = ctx.output_dir / "volcano.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def plot_heatmap(ctx: ToolContext, params: dict[str, Any]) -> str:
    genes = list(params.get("genes", []))
    if not genes:
        var = ctx.expression.var(axis=1).sort_values(ascending=False).head(30)
        genes = list(var.index.astype(str))
    sub = ctx.expression.loc[genes].astype(float)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sub.values, aspect="auto", cmap="vlag", vmin=-3, vmax=3)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=6)
    ax.set_xticks(range(sub.shape[1]))
    ax.set_xticklabels([str(c) for c in sub.columns], rotation=90, fontsize=6)
    ax.set_title("Expression heatmap (row z-score implicit via scaling in view)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = ctx.output_dir / "heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def plot_pca_scatter(ctx: ToolContext, params: dict[str, Any]) -> str:
    from src.tools.data_analysis import dimensionality_reduction

    res = dimensionality_reduction(ctx, {"n_components": 2})
    coords = np.array(res["coords"])
    labels = res["group_labels"]
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    uniq = sorted(set(labels))
    colors = ["#2563eb", "#16a34a", "#ca8a04", "#9333ea"]
    for i, u in enumerate(uniq):
        m = np.array([lab == u for lab in labels])
        ax.scatter(coords[m, 0], coords[m, 1], s=40, c=colors[i % len(colors)], label=u, alpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of samples")
    ax.legend(frameon=False)
    out = ctx.output_dir / "pca.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def plot_box_gene(ctx: ToolContext, params: dict[str, Any]) -> str:
    gene = str(params.get("gene", ctx.expression.index[0]))
    if gene not in ctx.expression.index:
        raise ValueError(f"Unknown gene {gene}")
    g1, g2 = sorted(ctx.groups.unique())
    a = ctx.expression.loc[gene, ctx.groups == g1].astype(float)
    b = ctx.expression.loc[gene, ctx.groups == g2].astype(float)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.boxplot([a.values, b.values], labels=[str(g1), str(g2)])
    ax.set_title(f"{gene} expression by group")
    out = ctx.output_dir / f"box_{gene}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def _tool_plot_volcano(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    path = plot_volcano(ctx, params)
    return {"figure_path": path}


def _tool_plot_heatmap(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    path = plot_heatmap(ctx, params)
    return {"figure_path": path}


def _tool_plot_pca(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    path = plot_pca_scatter(ctx, params)
    return {"figure_path": path}


def _tool_plot_box_gene(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    path = plot_box_gene(ctx, params)
    return {"figure_path": path}


def register_visualization_tools(registry: dict) -> None:
    registry["plot_volcano"] = _tool_plot_volcano
    registry["plot_heatmap"] = _tool_plot_heatmap
    registry["plot_pca"] = _tool_plot_pca
    registry["plot_box_gene"] = _tool_plot_box_gene
