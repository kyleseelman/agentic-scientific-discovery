"""Causal inference tools for the research agent.

Goes beyond correlation to estimate causal structure and effects from gene
expression data.  Implements PC algorithm, mediation analysis, instrumental
variables (2SLS), propensity-score methods, and regulatory network inference
without requiring specialised causal-inference packages — only numpy, scipy,
sklearn, and statsmodels.

All tools follow the TOOL_REGISTRY signature:
``(ctx: ToolContext, params: dict) -> dict``.
"""

from __future__ import annotations

import time
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.tools.data_analysis import ToolContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_output(ctx: ToolContext) -> Path:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    return ctx.output_dir


def _to_python(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _extract_expression_matrix(
    ctx: ToolContext,
    gene_subset: list[str] | None,
    n_top_genes: int = 50,
) -> tuple[np.ndarray, list[str]]:
    """Return (samples × genes) matrix and gene name list.

    Expression DataFrame is genes-by-samples; we transpose for analysis.
    """
    expr = ctx.expression
    if gene_subset:
        available = [g for g in gene_subset if g in expr.index]
        if not available:
            raise ValueError(
                f"None of the specified genes found. "
                f"Available (first 10): {list(expr.index[:10])}"
            )
        expr = expr.loc[available]
    else:
        var = expr.var(axis=1).sort_values(ascending=False)
        expr = expr.loc[var.head(n_top_genes).index]

    X = expr.T.astype(float).values  # samples × genes
    gene_names = list(expr.index.astype(str))
    return X, gene_names


def _encode_groups(ctx: ToolContext) -> np.ndarray:
    """Binary-encode the group labels (0/1)."""
    le = LabelEncoder()
    return le.fit_transform(ctx.groups.values)


# ---------------------------------------------------------------------------
# Partial correlation helpers (used by PC algorithm & network inference)
# ---------------------------------------------------------------------------

def _partial_correlation_matrix_gpu(X: np.ndarray) -> np.ndarray:
    """Compute partial correlation matrix via precision matrix (GPU path)."""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no GPU")

        device = torch.device("cuda")
        Xt = torch.tensor(X, dtype=torch.float64, device=device)
        Xt = Xt - Xt.mean(dim=0, keepdim=True)
        cov = (Xt.T @ Xt) / (Xt.shape[0] - 1)
        reg = 1e-6 * torch.eye(cov.shape[0], device=device, dtype=torch.float64)
        prec = torch.linalg.inv(cov + reg)
        d = torch.sqrt(torch.diag(prec))
        pcor = -(prec / (d[:, None] * d[None, :])).cpu().numpy()
        np.fill_diagonal(pcor, 1.0)
        return pcor
    except Exception:
        return _partial_correlation_matrix_cpu(X)


def _partial_correlation_matrix_cpu(X: np.ndarray) -> np.ndarray:
    """Compute partial correlation matrix via precision matrix (CPU path)."""
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    reg = 1e-6 * np.eye(cov.shape[0])
    prec = np.linalg.inv(cov + reg)
    d = np.sqrt(np.diag(prec))
    pcor = -(prec / np.outer(d, d))
    np.fill_diagonal(pcor, 1.0)
    return pcor


def _partial_corr_given_set(
    X: np.ndarray, i: int, j: int, cond_set: list[int],
) -> tuple[float, float]:
    """Partial correlation of columns *i* and *j* given *cond_set* via OLS.

    Returns (partial_r, p_value) using Fisher's z-test.
    """
    n = X.shape[0]
    if not cond_set:
        r, _ = stats.pearsonr(X[:, i], X[:, j])
    else:
        Z = X[:, cond_set]
        from numpy.linalg import lstsq
        # residualise i and j on Z
        beta_i, _, _, _ = lstsq(Z, X[:, i], rcond=None)
        beta_j, _, _, _ = lstsq(Z, X[:, j], rcond=None)
        res_i = X[:, i] - Z @ beta_i
        res_j = X[:, j] - Z @ beta_j
        r, _ = stats.pearsonr(res_i, res_j)

    # Fisher z-test
    dof = n - len(cond_set) - 2
    if dof < 1:
        return float(r), 1.0
    z = 0.5 * np.log((1 + r + 1e-15) / (1 - r + 1e-15))
    se = 1.0 / np.sqrt(max(dof - 1, 1))
    p = 2.0 * stats.norm.sf(abs(z) / se)
    return float(r), float(p)


# ---------------------------------------------------------------------------
# Tool 1: causal_graph_discovery
# ---------------------------------------------------------------------------

def causal_graph_discovery(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Discover causal graph structure from gene expression data.

    params:
        gene_subset: list[str] | None — genes to include (max ~50 for tractability)
        n_top_genes: int — fallback (default 30)
        method: str — "pc" (Peter-Clark), "ges" (Greedy Equivalence Search),
                       "lingam" (Linear Non-Gaussian), "granger" (Granger causality)
        alpha: float — significance threshold (default 0.05)
        max_conditioning_set: int — max set size for conditional independence (default 3)
    """
    gene_subset = params.get("gene_subset")
    n_top_genes = int(params.get("n_top_genes", 30))
    method = str(params.get("method", "pc"))
    alpha = float(params.get("alpha", 0.05))
    max_cond = int(params.get("max_conditioning_set", 3))

    X, gene_names = _extract_expression_matrix(ctx, gene_subset, n_top_genes)
    p = len(gene_names)

    print(f"Causal graph discovery ({method}) on {p} genes, "
          f"{X.shape[0]} samples, alpha={alpha}")
    t0 = time.time()

    if method == "pc":
        adj, edge_info = _pc_algorithm(X, alpha, max_cond)
    elif method == "ges":
        adj, edge_info = _ges_algorithm(X)
    elif method == "lingam":
        adj, edge_info = _lingam_algorithm(X)
    elif method == "granger":
        adj, edge_info = _granger_algorithm(X, alpha)
    else:
        return {"error": f"Unknown method '{method}'. "
                f"Choose: pc, ges, lingam, granger"}

    elapsed = time.time() - t0

    edges: list[dict[str, Any]] = []
    for (i, j), info in edge_info.items():
        edges.append({
            "source": gene_names[i],
            "target": gene_names[j],
            "weight": float(info.get("weight", abs(adj[i, j]))),
            "confidence": float(info.get("confidence", 1.0 - info.get("p_value", 0.0))),
        })

    out_degree = np.sum(adj != 0, axis=1)
    in_degree = np.sum(adj != 0, axis=0)
    total_degree = out_degree + in_degree

    top_out = np.argsort(-out_degree)[:10]
    top_in = np.argsort(-in_degree)[:10]

    potential_drivers = [
        {"gene": gene_names[i], "out_degree": int(out_degree[i]),
         "in_degree": int(in_degree[i])}
        for i in top_out if out_degree[i] > 0
    ]
    potential_targets = [
        {"gene": gene_names[i], "in_degree": int(in_degree[i]),
         "out_degree": int(out_degree[i])}
        for i in top_in if in_degree[i] > 0
    ]

    n_edges = int(np.sum(adj != 0))
    avg_degree = float(total_degree.mean()) if p > 0 else 0.0

    out_dir = _ensure_output(ctx)
    adj_df = pd.DataFrame(adj, index=gene_names, columns=gene_names)
    adj_path = out_dir / "causal_adjacency_matrix.csv"
    adj_df.to_csv(adj_path)

    print(f"Discovered {n_edges} directed edges in {elapsed:.1f}s")

    return _to_python({
        "method": method,
        "n_genes": p,
        "n_edges": n_edges,
        "avg_degree": avg_degree,
        "edges": edges,
        "adjacency_matrix_path": str(adj_path),
        "potential_drivers": potential_drivers,
        "potential_targets": potential_targets,
        "graph_stats": {
            "n_edges": n_edges,
            "avg_degree": avg_degree,
            "max_out_degree": int(out_degree.max()) if p > 0 else 0,
            "max_in_degree": int(in_degree.max()) if p > 0 else 0,
            "density": float(n_edges / (p * (p - 1))) if p > 1 else 0.0,
        },
        "alpha": alpha,
        "elapsed_s": elapsed,
    })


def _pc_algorithm(
    X: np.ndarray, alpha: float, max_cond: int,
) -> tuple[np.ndarray, dict[tuple[int, int], dict]]:
    """PC algorithm using conditional independence with Fisher z-test."""
    n, p = X.shape
    adj = np.ones((p, p), dtype=float)
    np.fill_diagonal(adj, 0)
    sep_sets: dict[tuple[int, int], list[int]] = {}
    edge_info: dict[tuple[int, int], dict] = {}

    # Phase 1: skeleton — remove edges via CI tests
    for cond_size in range(max_cond + 1):
        for i in range(p):
            for j in range(i + 1, p):
                if adj[i, j] == 0:
                    continue
                neighbours_i = [
                    k for k in range(p)
                    if k != i and k != j and adj[i, k] != 0
                ]
                if len(neighbours_i) < cond_size:
                    continue
                for cond in combinations(neighbours_i, cond_size):
                    r, pval = _partial_corr_given_set(X, i, j, list(cond))
                    if pval > alpha:
                        adj[i, j] = 0
                        adj[j, i] = 0
                        sep_sets[(i, j)] = list(cond)
                        sep_sets[(j, i)] = list(cond)
                        break

    # Phase 2: orient v-structures (i -> k <- j if k not in sep(i,j))
    oriented = np.zeros_like(adj)
    for i in range(p):
        for j in range(i + 1, p):
            if adj[i, j] != 0:
                continue
            common = [
                k for k in range(p)
                if adj[i, k] != 0 and adj[j, k] != 0
            ]
            for k in common:
                sep = sep_sets.get((i, j), [])
                if k not in sep:
                    oriented[i, k] = 1
                    oriented[j, k] = 1

    # Build directed adjacency
    directed = np.zeros((p, p), dtype=float)
    for i in range(p):
        for j in range(p):
            if adj[i, j] == 0:
                continue
            if oriented[i, j] == 1:
                directed[i, j] = 1
            elif oriented[j, i] == 1:
                pass  # j->i already handled
            else:
                r, pval = _partial_corr_given_set(X, i, j, [])
                directed[i, j] = abs(r)

    for i in range(p):
        for j in range(p):
            if directed[i, j] != 0:
                r, pval = _partial_corr_given_set(X, i, j, [])
                edge_info[(i, j)] = {
                    "weight": abs(r),
                    "p_value": pval,
                    "confidence": 1.0 - pval,
                }

    return directed, edge_info


def _ges_algorithm(
    X: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[int, int], dict]]:
    """Greedy Equivalence Search using BIC score."""
    n, p = X.shape
    adj = np.zeros((p, p), dtype=float)
    edge_info: dict[tuple[int, int], dict] = {}

    def _bic_local(y: np.ndarray, parents_idx: list[int]) -> float:
        if not parents_idx:
            rss = float(np.sum((y - y.mean()) ** 2))
        else:
            Z = X[:, parents_idx]
            beta, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
            rss = float(np.sum((y - Z @ beta) ** 2))
        rss = max(rss, 1e-15)
        k = len(parents_idx) + 1
        return n * np.log(rss / n) + k * np.log(n)

    # Forward phase: greedily add edges that lower BIC
    improved = True
    while improved:
        improved = False
        best_gain = 0.0
        best_edge = None
        for j in range(p):
            parents_j = list(np.where(adj[:, j] != 0)[0])
            bic_current = _bic_local(X[:, j], parents_j)
            for i in range(p):
                if i == j or adj[i, j] != 0:
                    continue
                bic_new = _bic_local(X[:, j], parents_j + [i])
                gain = bic_current - bic_new
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (i, j)

        if best_edge is not None and best_gain > 0:
            i, j = best_edge
            adj[i, j] = 1.0
            improved = True

    # Backward phase: remove edges that lower BIC
    improved = True
    while improved:
        improved = False
        best_gain = 0.0
        best_edge = None
        for j in range(p):
            parents_j = list(np.where(adj[:, j] != 0)[0])
            bic_current = _bic_local(X[:, j], parents_j)
            for i in parents_j:
                reduced = [pp for pp in parents_j if pp != i]
                bic_new = _bic_local(X[:, j], reduced)
                gain = bic_current - bic_new
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (i, j)

        if best_edge is not None and best_gain > 0:
            i, j = best_edge
            adj[i, j] = 0.0
            improved = True

    for i in range(p):
        for j in range(p):
            if adj[i, j] != 0:
                r, pval = stats.pearsonr(X[:, i], X[:, j])
                edge_info[(i, j)] = {
                    "weight": abs(r),
                    "p_value": pval,
                    "confidence": 1.0 - pval,
                }

    return adj, edge_info


def _lingam_algorithm(
    X: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[int, int], dict]]:
    """ICA-based LiNGAM using sklearn's FastICA."""
    from sklearn.decomposition import FastICA

    n, p = X.shape
    X_centered = X - X.mean(axis=0)

    ica = FastICA(n_components=p, random_state=42, max_iter=1000)
    try:
        S = ica.fit_transform(X_centered)
    except Exception:
        ica = FastICA(n_components=min(p, n - 1), random_state=42, max_iter=1000)
        S = ica.fit_transform(X_centered)

    W = ica.mixing_  # X ≈ S @ W^T => W is mixing matrix

    # Causal order: sort by row-norms of W^{-1} (diagonal dominance)
    try:
        W_inv = np.linalg.pinv(W)
    except np.linalg.LinAlgError:
        W_inv = np.linalg.pinv(W)

    # Permute to make W_inv as lower-triangular as possible
    order = np.argsort(np.abs(np.diag(W_inv)))[::-1]

    adj = np.zeros((p, p), dtype=float)
    edge_info: dict[tuple[int, int], dict] = {}

    # Regress each variable on those earlier in the causal order
    for idx in range(1, len(order)):
        j = order[idx]
        parents = order[:idx]
        Z = X_centered[:, parents]
        beta, _, _, _ = np.linalg.lstsq(Z, X_centered[:, j], rcond=None)
        for k, parent_idx in enumerate(parents):
            if abs(beta[k]) > 0.05:
                adj[parent_idx, j] = float(beta[k])
                r, pval = stats.pearsonr(X[:, parent_idx], X[:, j])
                edge_info[(parent_idx, j)] = {
                    "weight": abs(float(beta[k])),
                    "p_value": float(pval),
                    "confidence": 1.0 - float(pval),
                }

    return adj, edge_info


def _granger_algorithm(
    X: np.ndarray, alpha: float,
) -> tuple[np.ndarray, dict[tuple[int, int], dict]]:
    """Granger-causality using lagged regression (treats samples as time).

    Applicable when samples have a temporal ordering.
    """
    n, p = X.shape
    lag = 1
    adj = np.zeros((p, p), dtype=float)
    edge_info: dict[tuple[int, int], dict] = {}

    if n <= lag + 2:
        return adj, edge_info

    for j in range(p):
        y = X[lag:, j]
        y_lag = X[:-lag, j].reshape(-1, 1)
        n_eff = len(y)

        # Restricted model: y_t ~ y_{t-1}
        beta_r, _, _, _ = np.linalg.lstsq(y_lag, y, rcond=None)
        rss_r = float(np.sum((y - y_lag @ beta_r) ** 2))

        for i in range(p):
            if i == j:
                continue
            # Unrestricted model: y_t ~ y_{t-1} + x_{t-1}
            x_lag = X[:-lag, i].reshape(-1, 1)
            Z = np.hstack([y_lag, x_lag])
            beta_u, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
            rss_u = float(np.sum((y - Z @ beta_u) ** 2))

            df1 = 1
            df2 = n_eff - Z.shape[1] - 1
            if df2 < 1 or rss_u < 1e-15:
                continue

            f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
            p_value = float(stats.f.sf(f_stat, df1, df2))

            if p_value < alpha:
                adj[i, j] = float(f_stat)
                edge_info[(i, j)] = {
                    "weight": float(f_stat),
                    "p_value": p_value,
                    "confidence": 1.0 - p_value,
                }

    return adj, edge_info


# ---------------------------------------------------------------------------
# Tool 2: mediation_analysis
# ---------------------------------------------------------------------------

def mediation_analysis(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Test mediation: does gene M mediate the effect of group on outcome gene Y?

    Uses Baron & Kenny approach + Sobel test.

    params:
        outcome_gene: str — gene Y
        mediator_gene: str — potential mediator gene M
        include_covariates: list[str] | None — additional genes to control for
    """
    import statsmodels.api as sm

    outcome_gene = str(params.get("outcome_gene", ""))
    mediator_gene = str(params.get("mediator_gene", ""))
    covariate_genes: list[str] = list(params.get("include_covariates") or [])

    if not outcome_gene or not mediator_gene:
        return {"error": "Both outcome_gene and mediator_gene are required"}

    expr = ctx.expression
    for g in [outcome_gene, mediator_gene] + covariate_genes:
        if g not in expr.index:
            return {"error": f"Gene '{g}' not found in expression data"}

    treatment = _encode_groups(ctx).astype(float)
    Y = expr.loc[outcome_gene].astype(float).values
    M = expr.loc[mediator_gene].astype(float).values

    covariates = None
    if covariate_genes:
        covariates = expr.loc[covariate_genes].T.astype(float).values

    print(f"Mediation analysis: {outcome_gene} <- {mediator_gene} <- group")
    t0 = time.time()

    def _add_covariates(X_base: np.ndarray) -> np.ndarray:
        if covariates is not None:
            return np.hstack([X_base, covariates])
        return X_base

    # Step 1 (path c): Total effect — group -> Y
    X_c = sm.add_constant(_add_covariates(treatment.reshape(-1, 1)))
    model_c = sm.OLS(Y, X_c).fit()
    total_effect = float(model_c.params[1])
    total_p = float(model_c.pvalues[1])

    # Step 2 (path a): group -> M
    X_a = sm.add_constant(_add_covariates(treatment.reshape(-1, 1)))
    model_a = sm.OLS(M, X_a).fit()
    a_coef = float(model_a.params[1])
    a_se = float(model_a.bse[1])
    a_p = float(model_a.pvalues[1])

    # Step 3 (paths c' and b): group + M -> Y
    X_cb = sm.add_constant(
        _add_covariates(np.column_stack([treatment, M]))
    )
    model_cb = sm.OLS(Y, X_cb).fit()
    direct_effect = float(model_cb.params[1])  # c' (group controlling for M)
    b_coef = float(model_cb.params[2])          # b (M -> Y controlling for group)
    b_se = float(model_cb.bse[2])
    direct_p = float(model_cb.pvalues[1])

    # Indirect effect = a * b
    indirect_effect = a_coef * b_coef

    # Proportion mediated
    if abs(total_effect) > 1e-10:
        proportion_mediated = indirect_effect / total_effect
    else:
        proportion_mediated = 0.0
    proportion_mediated = float(np.clip(proportion_mediated, -1.0, 1.0))

    # Sobel test
    sobel_se = np.sqrt(a_coef**2 * b_se**2 + b_coef**2 * a_se**2)
    if sobel_se > 1e-15:
        sobel_z = indirect_effect / sobel_se
        sobel_p = float(2.0 * stats.norm.sf(abs(sobel_z)))
    else:
        sobel_z = 0.0
        sobel_p = 1.0

    elapsed = time.time() - t0
    is_sig = sobel_p < 0.05

    print(f"  Total={total_effect:.4f}, Direct={direct_effect:.4f}, "
          f"Indirect={indirect_effect:.4f}, Sobel p={sobel_p:.4g}, "
          f"mediation={'YES' if is_sig else 'NO'}")

    return _to_python({
        "outcome_gene": outcome_gene,
        "mediator_gene": mediator_gene,
        "total_effect": total_effect,
        "total_effect_p": total_p,
        "direct_effect": direct_effect,
        "direct_effect_p": direct_p,
        "indirect_effect": indirect_effect,
        "proportion_mediated": proportion_mediated,
        "path_a_coef": a_coef,
        "path_a_p": a_p,
        "path_b_coef": b_coef,
        "sobel_test_statistic": float(sobel_z),
        "sobel_p_value": sobel_p,
        "is_significant_mediation": is_sig,
        "covariates_controlled": covariate_genes,
        "elapsed_s": elapsed,
    })


# ---------------------------------------------------------------------------
# Tool 3: instrumental_variable_analysis
# ---------------------------------------------------------------------------

def instrumental_variable_analysis(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Instrumental variable (2SLS) analysis for causal effect estimation.

    params:
        outcome_gene: str — dependent variable
        treatment_gene: str — independent variable (potentially confounded)
        instrument_genes: list[str] — instruments (correlated with treatment,
                          not directly with outcome)
    """
    outcome_gene = str(params.get("outcome_gene", ""))
    treatment_gene = str(params.get("treatment_gene", ""))
    instrument_genes: list[str] = list(params.get("instrument_genes", []))

    if not outcome_gene or not treatment_gene or not instrument_genes:
        return {"error": "outcome_gene, treatment_gene, and instrument_genes "
                "are all required"}

    expr = ctx.expression
    all_genes = [outcome_gene, treatment_gene] + instrument_genes
    for g in all_genes:
        if g not in expr.index:
            return {"error": f"Gene '{g}' not found in expression data"}

    Y = expr.loc[outcome_gene].astype(float).values
    D = expr.loc[treatment_gene].astype(float).values
    Z = expr.loc[instrument_genes].T.astype(float).values  # samples × instruments
    n = len(Y)

    print(f"IV analysis: {outcome_gene} ~ {treatment_gene}, "
          f"instruments={instrument_genes}")
    t0 = time.time()

    # OLS estimate (naive)
    D_const = np.column_stack([np.ones(n), D])
    beta_ols, _, _, _ = np.linalg.lstsq(D_const, Y, rcond=None)
    ols_estimate = float(beta_ols[1])

    # First stage: D ~ Z
    Z_const = np.column_stack([np.ones(n), Z])
    beta_first, _, _, _ = np.linalg.lstsq(Z_const, D, rcond=None)
    D_hat = Z_const @ beta_first

    # First-stage F-statistic
    rss_restricted = float(np.sum((D - D.mean()) ** 2))
    rss_unrestricted = float(np.sum((D - D_hat) ** 2))
    k = Z.shape[1]
    df1 = k
    df2 = n - k - 1
    if df2 > 0 and rss_unrestricted > 1e-15:
        first_stage_f = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)
    else:
        first_stage_f = 0.0
    first_stage_f = float(first_stage_f)

    weak_instrument = first_stage_f < 10.0

    # Second stage: Y ~ D_hat
    D_hat_const = np.column_stack([np.ones(n), D_hat])
    beta_iv, _, _, _ = np.linalg.lstsq(D_hat_const, Y, rcond=None)
    iv_estimate = float(beta_iv[1])

    # IV standard error (using original residuals, not second-stage residuals)
    resid_iv = Y - np.column_stack([np.ones(n), D]) @ beta_iv
    sigma2 = float(np.sum(resid_iv ** 2) / max(n - 2, 1))
    DhDh = D_hat_const.T @ D_hat_const
    try:
        var_iv = sigma2 * np.linalg.inv(DhDh)
        iv_se = float(np.sqrt(var_iv[1, 1]))
    except np.linalg.LinAlgError:
        iv_se = float("inf")

    # Hausman test: compare OLS and IV
    diff = iv_estimate - ols_estimate
    resid_ols = Y - D_const @ beta_ols
    sigma2_ols = float(np.sum(resid_ols ** 2) / max(n - 2, 1))
    DtD = D_const.T @ D_const
    try:
        var_ols = sigma2_ols * np.linalg.inv(DtD)
        ols_se = float(np.sqrt(var_ols[1, 1]))
    except np.linalg.LinAlgError:
        ols_se = float("inf")

    hausman_var = max(iv_se**2 - ols_se**2, 1e-15)
    hausman_stat = diff**2 / hausman_var
    hausman_p = float(stats.chi2.sf(hausman_stat, df=1))

    elapsed = time.time() - t0

    # Instrument relevance check (correlation with treatment)
    instrument_corrs = []
    for i, ig in enumerate(instrument_genes):
        r, p_val = stats.pearsonr(Z[:, i], D)
        instrument_corrs.append({
            "instrument": ig,
            "corr_with_treatment": float(r),
            "p_value": float(p_val),
        })

    print(f"  OLS={ols_estimate:.4f}, IV={iv_estimate:.4f}, "
          f"F={first_stage_f:.1f}, Hausman p={hausman_p:.4g}")

    return _to_python({
        "outcome_gene": outcome_gene,
        "treatment_gene": treatment_gene,
        "instrument_genes": instrument_genes,
        "ols_estimate": ols_estimate,
        "ols_se": ols_se,
        "iv_estimate": iv_estimate,
        "iv_se": iv_se,
        "first_stage_f": first_stage_f,
        "weak_instrument_warning": weak_instrument,
        "hausman_test_statistic": float(hausman_stat),
        "hausman_test_p": hausman_p,
        "instrument_correlations": instrument_corrs,
        "n_samples": n,
        "elapsed_s": elapsed,
    })


# ---------------------------------------------------------------------------
# Tool 4: counterfactual_analysis
# ---------------------------------------------------------------------------

def counterfactual_analysis(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Estimate causal effects using propensity score matching/weighting.

    params:
        outcome_gene: str — gene to measure effect on
        covariate_genes: list[str] | None — genes to balance between groups
        n_top_genes: int — number of covariates if not specified (default 50)
        method: str — "matching", "iptw" (inverse propensity weighting),
                      "doubly_robust"
    """
    from sklearn.linear_model import LogisticRegression

    outcome_gene = str(params.get("outcome_gene", ""))
    covariate_genes: list[str] | None = params.get("covariate_genes")
    n_top_genes = int(params.get("n_top_genes", 50))
    method = str(params.get("method", "iptw"))

    if not outcome_gene:
        return {"error": "outcome_gene is required"}

    expr = ctx.expression
    if outcome_gene not in expr.index:
        return {"error": f"Gene '{outcome_gene}' not found in expression data"}

    # Get covariates (excluding outcome)
    if covariate_genes:
        cov_genes = [g for g in covariate_genes if g in expr.index and g != outcome_gene]
    else:
        var = expr.var(axis=1).sort_values(ascending=False)
        cov_genes = [g for g in var.head(n_top_genes).index if g != outcome_gene]

    if len(cov_genes) < 2:
        return {"error": "Need at least 2 covariate genes for propensity scoring"}

    X_cov = expr.loc[cov_genes].T.astype(float).values
    X_scaled = StandardScaler().fit_transform(X_cov)
    Y = expr.loc[outcome_gene].astype(float).values
    treatment = _encode_groups(ctx).astype(float)
    n = len(treatment)

    print(f"Counterfactual analysis ({method}): outcome={outcome_gene}, "
          f"{len(cov_genes)} covariates")
    t0 = time.time()

    # Fit propensity score model
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X_scaled, treatment)
    ps = ps_model.predict_proba(X_scaled)[:, 1]
    ps = np.clip(ps, 0.01, 0.99)

    # Balance diagnostics before adjustment
    treated = treatment == 1
    control = treatment == 0
    balance_before = {}
    for i, g in enumerate(cov_genes[:20]):
        d = (X_cov[treated, i].mean() - X_cov[control, i].mean())
        pooled_sd = np.sqrt(
            (X_cov[treated, i].var() + X_cov[control, i].var()) / 2 + 1e-15
        )
        balance_before[g] = float(d / pooled_sd)

    if method == "matching":
        ate, att, ate_ci, balance_after = _propensity_matching(
            Y, treatment, ps, X_cov, cov_genes, treated, control,
        )
    elif method == "iptw":
        ate, att, ate_ci, balance_after = _iptw(
            Y, treatment, ps, X_cov, cov_genes, treated, control,
        )
    elif method == "doubly_robust":
        ate, att, ate_ci, balance_after = _doubly_robust(
            Y, treatment, ps, X_scaled, X_cov, cov_genes, treated, control,
        )
    else:
        return {"error": f"Unknown method '{method}'. "
                f"Choose: matching, iptw, doubly_robust"}

    elapsed = time.time() - t0

    ps_summary = {
        "mean_treated": float(ps[treated].mean()),
        "mean_control": float(ps[control].mean()),
        "std_treated": float(ps[treated].std()),
        "std_control": float(ps[control].std()),
        "min": float(ps.min()),
        "max": float(ps.max()),
    }

    smd_before = np.mean([abs(v) for v in balance_before.values()])
    smd_after = np.mean([abs(v) for v in balance_after.values()])

    print(f"  ATE={ate:.4f}, ATT={att:.4f}, CI=({ate_ci[0]:.4f}, {ate_ci[1]:.4f})")
    print(f"  Balance SMD: {smd_before:.3f} -> {smd_after:.3f}")

    return _to_python({
        "outcome_gene": outcome_gene,
        "method": method,
        "n_covariates": len(cov_genes),
        "ate": ate,
        "att": att,
        "ate_ci": list(ate_ci),
        "balance_before": balance_before,
        "balance_after": balance_after,
        "mean_smd_before": smd_before,
        "mean_smd_after": smd_after,
        "propensity_scores": ps_summary,
        "n_treated": int(treated.sum()),
        "n_control": int(control.sum()),
        "elapsed_s": elapsed,
    })


def _propensity_matching(
    Y: np.ndarray,
    treatment: np.ndarray,
    ps: np.ndarray,
    X_cov: np.ndarray,
    cov_genes: list[str],
    treated: np.ndarray,
    control: np.ndarray,
) -> tuple[float, float, tuple[float, float], dict]:
    """Nearest-neighbour propensity-score matching."""
    from scipy.spatial import KDTree

    ps_treated = ps[treated].reshape(-1, 1)
    ps_control = ps[control].reshape(-1, 1)
    idx_treated = np.where(treated)[0]
    idx_control = np.where(control)[0]

    tree = KDTree(ps_control)
    _, nn_idx = tree.query(ps_treated, k=1)
    nn_idx = nn_idx.flatten()

    matched_treated_Y = Y[idx_treated]
    matched_control_Y = Y[idx_control[nn_idx]]

    att = float(np.mean(matched_treated_Y - matched_control_Y))
    ate = att  # with 1:1 matching, ATE ≈ ATT

    diffs = matched_treated_Y - matched_control_Y
    se = float(np.std(diffs) / np.sqrt(len(diffs)))
    ate_ci = (ate - 1.96 * se, ate + 1.96 * se)

    # Balance after matching
    balance_after: dict[str, float] = {}
    for i, g in enumerate(cov_genes[:20]):
        d = (X_cov[idx_treated, i].mean() - X_cov[idx_control[nn_idx], i].mean())
        pooled_sd = np.sqrt(
            (X_cov[idx_treated, i].var() + X_cov[idx_control[nn_idx], i].var()) / 2
            + 1e-15
        )
        balance_after[g] = float(d / pooled_sd)

    return ate, att, ate_ci, balance_after


def _iptw(
    Y: np.ndarray,
    treatment: np.ndarray,
    ps: np.ndarray,
    X_cov: np.ndarray,
    cov_genes: list[str],
    treated: np.ndarray,
    control: np.ndarray,
) -> tuple[float, float, tuple[float, float], dict]:
    """Inverse Probability of Treatment Weighting."""
    w1 = treatment / ps
    w0 = (1 - treatment) / (1 - ps)

    ate = float(np.mean(w1 * Y) - np.mean(w0 * Y))

    # ATT: weight control to look like treated
    w_att = ps / (1 - ps)
    att = float(
        np.mean(Y[treated]) -
        np.sum(w_att[control] * Y[control]) / np.sum(w_att[control])
    )

    # Bootstrap CI
    rng = np.random.RandomState(42)
    n = len(Y)
    boot_ates: list[float] = []
    for _ in range(200):
        idx = rng.choice(n, size=n, replace=True)
        w1b = treatment[idx] / ps[idx]
        w0b = (1 - treatment[idx]) / (1 - ps[idx])
        boot_ates.append(float(np.mean(w1b * Y[idx]) - np.mean(w0b * Y[idx])))
    ate_ci = (float(np.percentile(boot_ates, 2.5)),
              float(np.percentile(boot_ates, 97.5)))

    # Weighted balance
    balance_after: dict[str, float] = {}
    for i, g in enumerate(cov_genes[:20]):
        wm_t = np.average(X_cov[treated, i], weights=w1[treated])
        wm_c = np.average(X_cov[control, i], weights=w0[control])
        pooled_sd = np.sqrt(
            (X_cov[treated, i].var() + X_cov[control, i].var()) / 2 + 1e-15
        )
        balance_after[g] = float((wm_t - wm_c) / pooled_sd)

    return ate, att, ate_ci, balance_after


def _doubly_robust(
    Y: np.ndarray,
    treatment: np.ndarray,
    ps: np.ndarray,
    X_scaled: np.ndarray,
    X_cov: np.ndarray,
    cov_genes: list[str],
    treated: np.ndarray,
    control: np.ndarray,
) -> tuple[float, float, tuple[float, float], dict]:
    """Doubly robust estimator combining outcome model and propensity score."""
    from sklearn.linear_model import Ridge

    # Outcome models for each group
    model_t = Ridge(alpha=1.0).fit(X_scaled[treated], Y[treated])
    model_c = Ridge(alpha=1.0).fit(X_scaled[control], Y[control])

    mu1 = model_t.predict(X_scaled)
    mu0 = model_c.predict(X_scaled)

    n = len(Y)
    dr1 = mu1 + treatment * (Y - mu1) / ps
    dr0 = mu0 + (1 - treatment) * (Y - mu0) / (1 - ps)

    ate = float(np.mean(dr1 - dr0))

    # ATT
    att = float(
        np.mean(treatment * (Y - mu0)) /
        max(np.mean(treatment), 1e-10) -
        np.mean((1 - treatment) * ps * (Y - mu0) / ((1 - ps) + 1e-10)) /
        max(np.mean(treatment), 1e-10)
    )

    # Bootstrap CI
    rng = np.random.RandomState(42)
    boot_ates: list[float] = []
    for _ in range(200):
        idx = rng.choice(n, size=n, replace=True)
        boot_ates.append(float(np.mean(dr1[idx] - dr0[idx])))
    ate_ci = (float(np.percentile(boot_ates, 2.5)),
              float(np.percentile(boot_ates, 97.5)))

    balance_after: dict[str, float] = {}
    for i, g in enumerate(cov_genes[:20]):
        w1 = treatment / ps
        w0 = (1 - treatment) / (1 - ps)
        wm_t = np.average(X_cov[treated, i], weights=w1[treated])
        wm_c = np.average(X_cov[control, i], weights=w0[control])
        pooled_sd = np.sqrt(
            (X_cov[treated, i].var() + X_cov[control, i].var()) / 2 + 1e-15
        )
        balance_after[g] = float((wm_t - wm_c) / pooled_sd)

    return ate, att, ate_ci, balance_after


# ---------------------------------------------------------------------------
# Tool 5: interaction_network_analysis
# ---------------------------------------------------------------------------

def interaction_network_analysis(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Infer regulatory relationships between genes using mutual information
    and conditional independence.

    params:
        gene_subset: list[str] | None
        n_top_genes: int — (default 50)
        method: str — "mutual_information", "partial_correlation", "aracne"
        threshold: float — edge significance threshold (default 0.05)
    """
    gene_subset = params.get("gene_subset")
    n_top_genes = int(params.get("n_top_genes", 50))
    method = str(params.get("method", "mutual_information"))
    threshold = float(params.get("threshold", 0.05))

    X, gene_names = _extract_expression_matrix(ctx, gene_subset, n_top_genes)
    p = len(gene_names)

    print(f"Network inference ({method}) on {p} genes, "
          f"{X.shape[0]} samples")
    t0 = time.time()

    if method == "mutual_information":
        adj, edge_info = _mi_network(X, threshold)
    elif method == "partial_correlation":
        adj, edge_info = _partial_corr_network(X, threshold)
    elif method == "aracne":
        adj, edge_info = _aracne_network(X, threshold)
    else:
        return {"error": f"Unknown method '{method}'. "
                f"Choose: mutual_information, partial_correlation, aracne"}

    elapsed = time.time() - t0

    edges: list[dict[str, Any]] = []
    for (i, j), info in edge_info.items():
        edges.append({
            "source": gene_names[i],
            "target": gene_names[j],
            "weight": float(info["weight"]),
            "type": info.get("type", "undirected"),
        })

    degree = np.sum(adj != 0, axis=0) + np.sum(adj != 0, axis=1)
    top_hub_idx = np.argsort(-degree)[:10]
    hub_genes = [
        {"gene": gene_names[i], "degree": int(degree[i])}
        for i in top_hub_idx if degree[i] > 0
    ]

    n_edges = len(edges)
    avg_degree = float(degree.mean()) if p > 0 else 0.0

    out_dir = _ensure_output(ctx)
    adj_df = pd.DataFrame(adj, index=gene_names, columns=gene_names)
    adj_path = out_dir / "interaction_network_adjacency.csv"
    adj_df.to_csv(adj_path)

    print(f"Inferred {n_edges} edges, top hub: "
          f"{hub_genes[0]['gene'] if hub_genes else 'none'}")

    return _to_python({
        "method": method,
        "n_genes": p,
        "n_edges": n_edges,
        "edges": edges,
        "hub_genes": hub_genes,
        "network_stats": {
            "n_edges": n_edges,
            "avg_degree": avg_degree,
            "max_degree": int(degree.max()) if p > 0 else 0,
            "density": float(n_edges / (p * (p - 1) / 2)) if p > 1 else 0.0,
        },
        "adjacency_matrix_path": str(adj_path),
        "threshold": threshold,
        "elapsed_s": elapsed,
    })


def _mi_network(
    X: np.ndarray, threshold: float,
) -> tuple[np.ndarray, dict[tuple[int, int], dict]]:
    """Mutual information network with permutation-based thresholding."""
    from sklearn.feature_selection import mutual_info_regression

    n, p = X.shape
    adj = np.zeros((p, p), dtype=float)
    edge_info: dict[tuple[int, int], dict] = {}

    for j in range(p):
        mi_scores = mutual_info_regression(
            X, X[:, j], random_state=42, n_neighbors=3,
        )
        mi_scores[j] = 0.0  # self

        # Permutation null to get threshold
        rng = np.random.RandomState(42 + j)
        null_max = []
        for _ in range(50):
            perm = rng.permutation(X[:, j])
            null_mi = mutual_info_regression(
                X, perm, random_state=42, n_neighbors=3,
            )
            null_max.append(null_mi.max())
        mi_threshold = np.percentile(null_max, 100 * (1 - threshold))

        for i in range(p):
            if i != j and mi_scores[i] > mi_threshold:
                adj[i, j] = mi_scores[i]
                key = (min(i, j), max(i, j))
                if key not in edge_info:
                    edge_info[key] = {
                        "weight": float(mi_scores[i]),
                        "type": "undirected",
                    }

    return adj, edge_info


def _partial_corr_network(
    X: np.ndarray, threshold: float,
) -> tuple[np.ndarray, dict[tuple[int, int], dict]]:
    """Network based on partial correlations via precision matrix."""
    pcor = _partial_correlation_matrix_gpu(X)
    n, p = X.shape

    adj = np.zeros((p, p), dtype=float)
    edge_info: dict[tuple[int, int], dict] = {}

    for i in range(p):
        for j in range(i + 1, p):
            r = pcor[i, j]
            # Fisher z-test for significance
            dof = max(n - p - 2, 1)
            z = 0.5 * np.log((1 + abs(r) + 1e-15) / (1 - abs(r) + 1e-15))
            se = 1.0 / np.sqrt(max(dof - 1, 1))
            p_val = 2.0 * stats.norm.sf(abs(z) / se)
            if p_val < threshold:
                adj[i, j] = r
                adj[j, i] = r
                edge_info[(i, j)] = {
                    "weight": abs(float(r)),
                    "p_value": float(p_val),
                    "type": "undirected",
                }

    return adj, edge_info


def _aracne_network(
    X: np.ndarray, threshold: float,
) -> tuple[np.ndarray, dict[tuple[int, int], dict]]:
    """ARACNe: MI network with Data Processing Inequality (DPI) pruning."""
    from sklearn.feature_selection import mutual_info_regression

    n, p = X.shape

    # Step 1: compute full MI matrix
    mi_matrix = np.zeros((p, p), dtype=float)
    for j in range(p):
        mi_scores = mutual_info_regression(
            X, X[:, j], random_state=42, n_neighbors=3,
        )
        mi_scores[j] = 0.0
        mi_matrix[:, j] = mi_scores

    # Symmetrize
    mi_matrix = (mi_matrix + mi_matrix.T) / 2

    # Step 2: DPI pruning — for each triple (i,j,k), remove the weakest edge
    adj = mi_matrix.copy()
    for i in range(p):
        for j in range(i + 1, p):
            if adj[i, j] <= 0:
                continue
            for k in range(p):
                if k == i or k == j:
                    continue
                if adj[i, j] < min(adj[i, k], adj[j, k]):
                    adj[i, j] = 0
                    adj[j, i] = 0
                    break

    # Step 3: threshold remaining edges
    nonzero = adj[adj > 0]
    if len(nonzero) > 0:
        cutoff = np.percentile(nonzero, threshold * 100)
        adj[adj < cutoff] = 0

    edge_info: dict[tuple[int, int], dict] = {}
    for i in range(p):
        for j in range(i + 1, p):
            if adj[i, j] > 0:
                edge_info[(i, j)] = {
                    "weight": float(adj[i, j]),
                    "type": "undirected",
                }

    return adj, edge_info


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_causal_tools(registry: dict[str, Callable]) -> None:
    """Register all causal inference tools into the TOOL_REGISTRY."""

    tools: dict[str, Callable] = {
        "causal_graph_discovery": causal_graph_discovery,
        "mediation_analysis": mediation_analysis,
        "instrumental_variable_analysis": instrumental_variable_analysis,
        "counterfactual_analysis": counterfactual_analysis,
        "interaction_network_analysis": interaction_network_analysis,
    }

    for name, fn in tools.items():
        def _make_safe(tool_fn: Callable, tool_name: str) -> Callable:
            def safe_wrapper(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
                try:
                    return tool_fn(ctx, params)
                except Exception as e:
                    import traceback
                    return {
                        "error": f"{tool_name} failed: {type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                    }
            safe_wrapper.__name__ = tool_name
            safe_wrapper.__doc__ = tool_fn.__doc__
            return safe_wrapper

        registry[name] = _make_safe(fn, name)

    print(f"Registered {len(tools)} causal inference tools: {', '.join(tools)}")
