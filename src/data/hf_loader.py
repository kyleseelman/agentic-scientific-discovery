"""HuggingFace Dataset Loader for omics data.

Uses the `datasets` library to load processed omics datasets from
HuggingFace Hub and normalizes them into ToolContext-compatible format.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def load_hf_omics_dataset(
    dataset_id: str,
    config: str | None = None,
    split: str = "train",
    gene_columns: list[str] | None = None,
    sample_id_column: str | None = None,
    group_column: str | None = None,
    control_label: str | None = None,
    treatment_label: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load a HuggingFace dataset and normalize to expression + groups format.

    Parameters
    ----------
    dataset_id : str
        HuggingFace dataset identifier (e.g. "user/dataset-name").
    config : str, optional
        Dataset configuration/subset name.
    split : str
        Which split to load (default "train").
    gene_columns : list of str, optional
        Column names representing gene expression values. If None, auto-detects
        numeric columns as gene features.
    sample_id_column : str, optional
        Column to use as sample identifier. If None, uses the index.
    group_column : str, optional
        Column containing group/condition labels. If None, attempts to detect
        from common column names.
    control_label : str, optional
        Value in group_column to treat as control.
    treatment_label : str, optional
        Value in group_column to treat as treatment.

    Returns
    -------
    tuple of (expression, groups, metadata)
        - expression: pd.DataFrame, genes (rows) x samples (columns)
        - groups: pd.Series, indexed by sample ID with group labels
        - metadata: dict with dataset info
    """
    from datasets import load_dataset

    logger.info("Loading HuggingFace dataset: %s (config=%s, split=%s)", dataset_id, config, split)

    kwargs: dict[str, Any] = {"path": dataset_id, "split": split}
    if config:
        kwargs["name"] = config

    ds = load_dataset(**kwargs)
    df = ds.to_pandas()

    logger.info("Dataset shape: %d rows x %d columns", *df.shape)

    if sample_id_column and sample_id_column in df.columns:
        df = df.set_index(sample_id_column)
    elif sample_id_column is None:
        id_candidates = ["sample_id", "Sample_ID", "id", "ID", "barcode", "sample"]
        for col in id_candidates:
            if col in df.columns:
                df = df.set_index(col)
                break

    if group_column is None:
        group_column = _detect_group_column(df)

    if group_column is None:
        raise ValueError(
            "Could not detect a group column. Available columns: "
            f"{list(df.columns[:20])}... "
            "Specify group_column explicitly."
        )

    groups = df[group_column].astype(str)

    if control_label and treatment_label:
        label_map = {control_label: "control", treatment_label: "treatment"}
        groups = groups.map(lambda x: label_map.get(x, x))
        mask = groups.isin(["control", "treatment"])
        groups = groups[mask]
        df = df.loc[groups.index]

    if gene_columns is None:
        gene_columns = _detect_gene_columns(df, exclude=[group_column])

    if not gene_columns:
        raise ValueError(
            "No gene/feature columns detected. Specify gene_columns explicitly."
        )

    expression = df[gene_columns].T.astype(float)
    expression.index.name = "gene"
    expression.columns = [str(s) for s in expression.columns]
    groups.index = [str(s) for s in groups.index]

    metadata = {
        "dataset_id": dataset_id,
        "config": config,
        "split": split,
        "group_column": group_column,
        "group_values": sorted(groups.unique().tolist()),
        "n_samples": len(groups),
        "n_genes": len(expression),
    }

    logger.info(
        "Loaded %s: %d genes x %d samples, groups: %s",
        dataset_id,
        len(expression),
        len(groups),
        metadata["group_values"],
    )
    return expression, groups, metadata


def _detect_group_column(df: pd.DataFrame) -> str | None:
    """Detect the most likely group/label column."""
    candidates = [
        "label", "group", "condition", "class", "target",
        "disease", "status", "phenotype", "diagnosis",
        "treatment", "response", "subtype", "category",
    ]

    for col in candidates:
        for df_col in df.columns:
            if df_col.lower() == col or df_col.lower().replace("_", " ") == col:
                unique = df[df_col].nunique()
                if 2 <= unique <= 20:
                    return df_col

    object_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in object_cols:
        unique = df[col].nunique()
        if 2 <= unique <= 10:
            return col

    return None


def _detect_gene_columns(
    df: pd.DataFrame,
    exclude: list[str] | None = None,
) -> list[str]:
    """Detect columns that likely represent gene expression values."""
    exclude_set = set(exclude or [])

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    gene_cols = [c for c in numeric_cols if c not in exclude_set]

    if len(gene_cols) > 50:
        return gene_cols

    gene_like = [
        c for c in gene_cols
        if any(c.upper().startswith(p) for p in ["ENSG", "GENE", "HG", "NM_"])
        or c.isupper()
    ]
    return gene_like if gene_like else gene_cols


def load_expression_csv(
    filepath: str,
    group_column: str | None = None,
    transpose: bool = False,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load expression data from a local CSV/TSV file.

    Convenience adapter for tabular files following common formats:
    - Genes as rows, samples as columns (default)
    - Samples as rows, genes as columns (transpose=True)

    Parameters
    ----------
    filepath : str
        Path to CSV or TSV file.
    group_column : str, optional
        Column in the file with group labels. If the file has a separate
        metadata structure, this tries to extract from it.
    transpose : bool
        If True, treats rows as samples and columns as genes.

    Returns
    -------
    tuple of (expression, groups, metadata)
    """
    sep = "\t" if filepath.endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(filepath, sep=sep, index_col=0)

    if transpose:
        expression = df.select_dtypes(include=["number"]).T
    else:
        expression = df.select_dtypes(include=["number"])

    expression.index.name = "gene"

    if group_column and group_column in df.columns:
        groups = df[group_column].astype(str)
    elif group_column and not transpose:
        raise ValueError(f"Group column '{group_column}' not found in file columns.")
    else:
        groups = pd.Series("unknown", index=expression.columns)

    metadata = {
        "source": filepath,
        "n_genes": len(expression),
        "n_samples": expression.shape[1],
        "group_column": group_column,
    }

    return expression, groups, metadata
