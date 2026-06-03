"""
Synthetic bulk RNA-seq–like expression matrix with planted pathway signal.

The agent can be evaluated on whether enrichment recovers `PLANTED_STRESS`
(or similar) using the same statistical tools as for real cohorts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_synthetic_cohort(
    n_genes: int = 800,
    n_control: int = 12,
    n_treat: int = 12,
    pathway_size: int = 45,
    effect: float = 1.35,
    noise: float = 0.85,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]]]:
    rng = np.random.default_rng(seed)
    genes = [f"GENE_{i:04d}" for i in range(n_genes)]
    pathway = genes[:pathway_size]
    decoy = genes[pathway_size : pathway_size + 40]

    controls = [f"CTRL_{i:02d}" for i in range(n_control)]
    treated = [f"TX_{i:02d}" for i in range(n_treat)]
    samples = controls + treated

    base = rng.normal(6.0, noise, size=(n_genes, len(samples)))

    pathway_mask = np.array([g in pathway for g in genes])
    base[np.ix_(pathway_mask, range(len(controls), len(samples)))] += effect + rng.normal(
        0, 0.15, size=(pathway_mask.sum(), n_treat)
    )
    base[np.ix_(np.array([g in decoy for g in genes]), range(0, len(controls)))] += rng.normal(
        0.2, 0.25, size=(len(decoy), n_control)
    )

    expr = pd.DataFrame(base, index=genes, columns=samples)
    groups = pd.Series(
        ["control"] * len(controls) + ["treatment"] * len(treated),
        index=samples,
    )
    pathway_sets = {
        "PLANTED_STRESS": pathway,
        "DECOY_NOISE_SET": decoy,
        "RANDOM_GO_001": list(rng.choice(genes, size=30, replace=False)),
        "RANDOM_GO_002": list(rng.choice(genes, size=30, replace=False)),
    }
    return expr, groups, pathway_sets


def write_example_artifacts(
    out_dir: Path,
    n_genes: int = 800,
    seed: int = 42,
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    expr, groups, pathways = build_synthetic_cohort(n_genes=n_genes, seed=seed)
    expr_path = out_dir / "expression.tsv"
    groups_path = out_dir / "groups.tsv"
    meta_path = out_dir / "ground_truth_pathways.json"
    expr.to_csv(expr_path, sep="\t")
    groups.to_frame(name="group").to_csv(groups_path, sep="\t")
    import json

    meta_path.write_text(json.dumps({"planted_pathways": pathways}, indent=2))
    return expr_path, groups_path, meta_path
