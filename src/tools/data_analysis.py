from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


@dataclass
class ToolContext:
    expression: pd.DataFrame
    groups: pd.Series
    pathway_sets: dict[str, list[str]] = field(default_factory=dict)
    gene_metadata: pd.DataFrame | None = None
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))


def _ensure_output(ctx: ToolContext) -> Path:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    return ctx.output_dir


def profile_dataset(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    g1, g2 = sorted(ctx.groups.unique())
    mask1 = ctx.groups == g1
    mask2 = ctx.groups == g2
    return {
        "n_samples": int(ctx.expression.shape[1]),
        "n_genes": int(ctx.expression.shape[0]),
        "group_counts": {str(k): int(v) for k, v in ctx.groups.value_counts().items()},
        "missing_fraction": float(ctx.expression.isna().mean().mean()),
        "mean_expression_std_across_samples": float(ctx.expression.std(axis=1).mean()),
        "top_variable_genes_head": list(ctx.expression.var(axis=1).sort_values(ascending=False).head(5).index.astype(str)),
        "groups_compared": [str(g1), str(g2)],
    }


def _gpu_welch_t(x1_np: np.ndarray, x2_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Welch's t-test on GPU using PyTorch. Returns (t_stats, p_values)."""
    import torch
    from scipy.stats import t as t_dist

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.tensor(x1_np, dtype=torch.float32, device=device)
    b = torch.tensor(x2_np, dtype=torch.float32, device=device)

    n1 = a.shape[1]
    n2 = b.shape[1]
    mean1 = a.mean(dim=1)
    mean2 = b.mean(dim=1)
    var1 = a.var(dim=1, unbiased=True)
    var2 = b.var(dim=1, unbiased=True)

    se = torch.sqrt(var1 / n1 + var2 / n2)
    t_stats = (mean1 - mean2) / se.clamp(min=1e-10)

    # Welch-Satterthwaite degrees of freedom
    num = (var1 / n1 + var2 / n2) ** 2
    denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    df = num / denom.clamp(min=1e-10)

    t_np = t_stats.cpu().numpy()
    df_np = df.cpu().numpy()
    logfc_np = (mean2 - mean1).cpu().numpy()

    p_values = 2 * t_dist.sf(np.abs(t_np), df_np)
    return t_np, p_values, logfc_np


def differential_expression(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    method = params.get("method", "welch_t")
    correction = params.get("correction", "fdr_bh")
    use_gpu = params.get("use_gpu", True)
    g1, g2 = sorted(ctx.groups.unique())
    x1 = ctx.expression.loc[:, ctx.groups == g1]
    x2 = ctx.expression.loc[:, ctx.groups == g2]

    gpu_used = False
    if use_gpu and method != "mannwhitney":
        try:
            import torch
            if torch.cuda.is_available():
                t_stats, p_values, logfc = _gpu_welch_t(x1.values.astype(float), x2.values.astype(float))
                rows = []
                for i, gene in enumerate(ctx.expression.index):
                    rows.append({
                        "gene": str(gene),
                        "log2_fold_change": float(logfc[i]),
                        "statistic": float(t_stats[i]) if not np.isnan(t_stats[i]) else 0.0,
                        "pvalue": float(p_values[i]) if p_values[i] == p_values[i] else 1.0,
                    })
                gpu_used = True
        except (ImportError, RuntimeError):
            pass

    if not gpu_used:
        rows = []
        for gene in ctx.expression.index:
            a = x1.loc[gene].astype(float).values
            b = x2.loc[gene].astype(float).values
            if method == "mannwhitney":
                stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                tstat = float(stat)
            else:
                tstat, p = stats.ttest_ind(a, b, equal_var=False)
                tstat = float(tstat) if not np.isnan(tstat) else 0.0
            logfc = float(np.mean(b) - np.mean(a))
            rows.append({
                "gene": str(gene),
                "log2_fold_change": logfc,
                "statistic": tstat,
                "pvalue": float(p) if p == p else 1.0,
            })

    de = pd.DataFrame(rows).set_index("gene")
    rej, q, _, _ = multipletests(de["pvalue"].values, method=correction)
    de["qvalue"] = q
    de["reject"] = rej
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    de_path = ctx.output_dir / "differential_expression.csv"
    de.to_csv(de_path)
    return {
        "table_path": str(de_path),
        "n_significant_q_0.05": int(de["reject"].sum()),
        "method": method,
        "correction": correction,
        "gpu_accelerated": gpu_used,
        "top_genes_by_q": de.sort_values("qvalue").head(20).reset_index().to_dict(orient="records"),
    }


def correlation_analysis(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    kind = params.get("kind", "spearman")
    genes: list[str] = list(params.get("genes", []))
    if len(genes) < 2:
        genes = list(ctx.expression.index[:5])
    sub = ctx.expression.loc[genes].astype(float).T
    mat = sub.corr(method="spearman" if kind == "spearman" else "pearson")
    return {
        "genes": genes,
        "correlation_matrix": mat.to_dict(),
        "kind": kind,
    }


def feature_importance_variance(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    top_n = int(params.get("top_n", 30))
    var = ctx.expression.var(axis=1).sort_values(ascending=False).head(top_n)
    g1, g2 = sorted(ctx.groups.unique())
    assoc = []
    for gene in var.index:
        v = ctx.expression.loc[gene].astype(float).values
        labels = (ctx.groups == g2).astype(int).values
        if len(set(labels)) < 2:
            continue
        r, p = spearmanr(v, labels)
        assoc.append({"gene": str(gene), "spearman_r": float(r), "pvalue": float(p)})
    return {"top_variable_genes": var.reset_index().to_dict(orient="records"), "group_association": assoc}


def dimensionality_reduction(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    n_comp = int(params.get("n_components", 2))
    X = ctx.expression.T.astype(float).values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_comp, random_state=0)
    coords = pca.fit_transform(Xs)
    out = {
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "coords": coords.tolist(),
        "sample_ids": [str(s) for s in ctx.expression.columns],
        "group_labels": [str(ctx.groups.loc[s]) for s in ctx.expression.columns],
    }
    _ensure_output(ctx)
    pd.DataFrame(coords, index=ctx.expression.columns, columns=[f"PC{i+1}" for i in range(n_comp)]).to_csv(
        ctx.output_dir / "pca_coords.csv"
    )
    return out


def clustering_samples(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    k = int(params.get("k", 3))
    X = ctx.expression.T.astype(float).values
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(Xs)
    return {
        "k": k,
        "labels": {str(sid): int(lab) for sid, lab in zip(ctx.expression.columns, labels)},
    }


def _hypergeom_sf(k: int, M: int, n: int, N: int) -> float:
    return float(stats.hypergeom.sf(k - 1, M, n, N))


def pathway_enrichment(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    de_path = ctx.output_dir / "differential_expression.csv"
    if not de_path.exists():
        raise FileNotFoundError("Run differential_expression before pathway_enrichment")
    de = pd.read_csv(de_path, index_col=0)
    q_thr = float(params.get("q_threshold", 0.05))
    direction = params.get("direction", "up")
    genes_all = set(ctx.expression.index.astype(str))
    if direction == "up":
        sig = de[(de["reject"]) & (de["log2_fold_change"] > 0)]
    elif direction == "down":
        sig = de[(de["reject"]) & (de["log2_fold_change"] < 0)]
    else:
        sig = de[de["reject"]]
    sig_genes = set(sig.index.astype(str))
    M = len(genes_all)
    N = len(sig_genes)
    results = []
    for pathway, members in ctx.pathway_sets.items():
        memb = [g for g in members if g in genes_all]
        n = len(memb)
        if n == 0:
            continue
        overlap = len(sig_genes.intersection(memb))
        if overlap == 0:
            p = 1.0
        else:
            p = _hypergeom_sf(overlap, M, n, N)
        results.append(
            {
                "pathway": pathway,
                "pathway_size": n,
                "overlap": overlap,
                "pvalue": p,
            }
        )
    df = pd.DataFrame(results)
    if df.empty:
        return {"enriched": [], "note": "no pathways defined or no overlap"}
    rej, q, _, _ = multipletests(df["pvalue"].clip(upper=1.0 - 1e-15), method="fdr_bh")
    df["qvalue"] = q
    df = df.sort_values("qvalue")
    enriched = df[df["qvalue"] <= q_thr].to_dict(orient="records")
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(ctx.output_dir / "pathway_enrichment.csv", index=False)
    return {
        "enriched": enriched,
        "direction": direction,
        "q_threshold": q_thr,
        "table_path": str(ctx.output_dir / "pathway_enrichment.csv"),
    }


def group_comparison_summary(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    genes: list[str] = list(params.get("genes", []))
    if not genes:
        genes = list(ctx.expression.index[:3])
    g1, g2 = sorted(ctx.groups.unique())
    out = []
    for gene in genes:
        if gene not in ctx.expression.index:
            continue
        a = ctx.expression.loc[gene, ctx.groups == g1].astype(float)
        b = ctx.expression.loc[gene, ctx.groups == g2].astype(float)
        stat, p = stats.ttest_ind(a, b, equal_var=False)
        out.append(
            {
                "gene": gene,
                "mean_control": float(a.mean()),
                "mean_treatment": float(b.mean()),
                "pvalue": float(p) if p == p else 1.0,
                "statistic": float(stat),
            }
        )
    return {"comparisons": out}


TOOL_REGISTRY: dict[str, Callable[[ToolContext, dict[str, Any]], dict[str, Any]]] = {
    "profile_dataset": profile_dataset,
    "differential_expression": differential_expression,
    "pathway_enrichment": pathway_enrichment,
    "correlation_analysis": correlation_analysis,
    "feature_importance_variance": feature_importance_variance,
    "dimensionality_reduction": dimensionality_reduction,
    "clustering_samples": clustering_samples,
    "group_comparison_summary": group_comparison_summary,
}


def _register_extended_tools() -> None:
    from src.tools import bio_databases
    from src.tools import literature as literature_tools
    from src.tools import visualization

    visualization.register_visualization_tools(TOOL_REGISTRY)
    bio_databases.register_bio_tools(TOOL_REGISTRY)

    def literature_pubmed(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        return literature_tools.search_pubmed(
            str(params.get("query", "gene expression biomarker")),
            retmax=int(params.get("retmax", 5)),
        )

    def literature_fetch_abstracts(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        pmids = [str(x) for x in params.get("pmids", [])]
        papers = literature_tools.fetch_pubmed_abstracts(pmids)
        return {
            "papers": [
                {"id": p.paper_id, "title": p.title, "abstract": p.abstract[:500], "journal": p.journal}
                for p in papers
            ]
        }

    def literature_search_biorxiv(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        papers = literature_tools.search_biorxiv(
            str(params.get("query", "")),
            max_results=int(params.get("max_results", 3)),
        )
        return {
            "papers": [
                {"id": p.paper_id, "title": p.title, "abstract": p.abstract[:500]}
                for p in papers
            ]
        }

    TOOL_REGISTRY["literature_pubmed"] = literature_pubmed
    TOOL_REGISTRY["literature_fetch_abstracts"] = literature_fetch_abstracts
    TOOL_REGISTRY["literature_search_biorxiv"] = literature_search_biorxiv

    from src.tools.code_executor import code_execution_tool
    TOOL_REGISTRY["execute_code"] = code_execution_tool

    from src.tools.knowledge_graph import register_knowledge_graph_tools
    register_knowledge_graph_tools(TOOL_REGISTRY)

    from src.tools.ml_models import register_ml_tools
    register_ml_tools(TOOL_REGISTRY)

    from src.tools.llm_tools import register_llm_tools
    register_llm_tools(TOOL_REGISTRY)

    from src.tools.causal_inference import register_causal_tools
    register_causal_tools(TOOL_REGISTRY)

    from src.tools.model_builder import register_model_builder_tools
    register_model_builder_tools(TOOL_REGISTRY)


_register_extended_tools()


def run_tool(
    name: str,
    ctx: ToolContext,
    params: dict[str, Any],
) -> dict[str, Any]:
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}")
    return TOOL_REGISTRY[name](ctx, params)
