"""GEO Expression Data Loader using GEOparse.

Downloads and parses GEO series into expression matrices and sample group labels
compatible with ToolContext (genes x samples DataFrame + sample-indexed Series).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "agentic-discovery" / "geo"

GROUP_FIELD_CANDIDATES = [
    "disease state",
    "disease_state",
    "treatment",
    "condition",
    "group",
    "phenotype",
    "status",
    "cell type",
    "cell_type",
    "tissue",
    "genotype",
    "time point",
    "time_point",
    "description",
]


def _get_cache_dir(cache_dir: Path | None = None) -> Path:
    d = cache_dir or DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_gse(accession: str, cache_dir: Path | None = None) -> Any:
    """Download/cache and return a GEOparse GSE object."""
    import GEOparse

    dest = _get_cache_dir(cache_dir)
    logger.info("Fetching %s (cache: %s)", accession, dest)
    gse = GEOparse.get_GEO(geo=accession, destdir=str(dest), silent=True)
    return gse


def list_sample_characteristics(
    accession: str,
    cache_dir: Path | None = None,
) -> dict[str, list[str]]:
    """Show available metadata fields and their unique values for a GEO series.

    Useful for deciding which column to use for grouping.
    Returns {field_name: [unique_values...]}.
    """
    gse = _parse_gse(accession, cache_dir)
    characteristics: dict[str, set[str]] = {}

    for gsm_name, gsm in gse.gsms.items():
        for key, values in gsm.metadata.items():
            if key.startswith("characteristics_ch"):
                for v in values:
                    if ": " in v:
                        field, val = v.split(": ", 1)
                        field = field.strip().lower()
                    else:
                        field = key
                        val = v.strip()
                    characteristics.setdefault(field, set()).add(val)

    return {k: sorted(v) for k, v in characteristics.items()}


def _detect_group_field(gse: Any) -> tuple[str, dict[str, str]]:
    """Auto-detect the best grouping field from sample characteristics.

    Returns (field_name, {sample_id: group_label}).
    """
    sample_chars: dict[str, dict[str, str]] = {}

    for gsm_name, gsm in gse.gsms.items():
        sample_chars[gsm_name] = {}
        for key, values in gsm.metadata.items():
            if key.startswith("characteristics_ch"):
                for v in values:
                    if ": " in v:
                        field, val = v.split(": ", 1)
                        field = field.strip().lower()
                        sample_chars[gsm_name][field] = val.strip()
                    else:
                        sample_chars[gsm_name][key] = v.strip()

    all_fields: set[str] = set()
    for chars in sample_chars.values():
        all_fields.update(chars.keys())

    for candidate in GROUP_FIELD_CANDIDATES:
        if candidate in all_fields:
            labels = {sid: chars.get(candidate, "unknown") for sid, chars in sample_chars.items()}
            unique_vals = set(labels.values()) - {"unknown"}
            if 2 <= len(unique_vals) <= 10:
                logger.info("Auto-detected group field: '%s' with values: %s", candidate, unique_vals)
                return candidate, labels

    for field in sorted(all_fields):
        labels = {sid: chars.get(field, "unknown") for sid, chars in sample_chars.items()}
        unique_vals = set(labels.values()) - {"unknown"}
        if 2 <= len(unique_vals) <= 10:
            logger.info("Fallback group field: '%s' with values: %s", field, unique_vals)
            return field, labels

    raise ValueError(
        f"Could not auto-detect a grouping field. Available fields: {sorted(all_fields)}. "
        "Use load_geo_dataset() with explicit group_column parameter."
    )


def _extract_expression(gse: Any) -> pd.DataFrame:
    """Extract the expression matrix (genes x samples) from a GSE object.

    Tries the GSE pivot table first, then falls back to individual GSM tables.
    """
    try:
        pivot = gse.pivot_samples("VALUE")
        if pivot is not None and not pivot.empty:
            pivot.index.name = "gene"
            pivot.columns.name = None
            return pivot
    except Exception:
        pass

    frames = {}
    for gsm_name, gsm in gse.gsms.items():
        tbl = gsm.table
        if tbl is not None and not tbl.empty:
            if "ID_REF" in tbl.columns and "VALUE" in tbl.columns:
                series = tbl.set_index("ID_REF")["VALUE"]
                series.index.name = "gene"
                frames[gsm_name] = series

    if not frames:
        raise ValueError(
            "No expression data found in this GEO series. "
            "It may use supplementary files instead of series matrix format."
        )

    expr = pd.DataFrame(frames)
    expr.index.name = "gene"
    return expr


def _map_probes_to_genes(expression: pd.DataFrame, gse: Any) -> pd.DataFrame:
    """Attempt to map probe IDs to gene symbols using the platform annotation."""
    gpls = list(gse.gpls.values())
    if not gpls:
        return expression

    gpl = gpls[0]
    ann = gpl.table

    if ann is None or ann.empty:
        return expression

    symbol_cols = [c for c in ann.columns if "symbol" in c.lower() or "gene_symbol" in c.lower()]
    if not symbol_cols:
        symbol_cols = [c for c in ann.columns if "gene" in c.lower() and "id" not in c.lower()]
    if not symbol_cols:
        return expression

    symbol_col = symbol_cols[0]
    id_col = "ID" if "ID" in ann.columns else ann.columns[0]

    probe_to_gene = ann.set_index(id_col)[symbol_col].dropna()
    probe_to_gene = probe_to_gene[probe_to_gene.str.strip() != ""]
    probe_to_gene = probe_to_gene[~probe_to_gene.str.contains("///", na=False)]

    common_probes = expression.index.intersection(probe_to_gene.index)
    if len(common_probes) < 10:
        return expression

    mapped = expression.loc[common_probes].copy()
    mapped.index = probe_to_gene.loc[common_probes].values
    mapped.index.name = "gene"
    mapped = mapped[~mapped.index.duplicated(keep="first")]
    mapped = mapped.loc[mapped.index.dropna()]
    mapped = mapped.loc[mapped.index != ""]

    logger.info(
        "Mapped %d probes to %d unique gene symbols (from %d total probes)",
        len(common_probes),
        len(mapped),
        len(expression),
    )
    return mapped


def load_geo_dataset(
    accession: str,
    group_column: str | None = None,
    control_label: str | None = None,
    treatment_label: str | None = None,
    map_to_genes: bool = True,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load a GEO dataset and return expression, groups, and metadata.

    Parameters
    ----------
    accession : str
        GEO series accession (e.g. "GSE130727").
    group_column : str, optional
        Specific metadata field to use for grouping. If None, auto-detects.
    control_label : str, optional
        Label to assign as "control". If None, keeps original labels.
    treatment_label : str, optional
        Label to assign as "treatment". If None, keeps original labels.
    map_to_genes : bool
        Whether to map probe IDs to gene symbols.
    cache_dir : Path, optional
        Cache directory for downloaded files.

    Returns
    -------
    tuple of (expression, groups, metadata)
        - expression: pd.DataFrame, genes (rows) x samples (columns)
        - groups: pd.Series, indexed by sample ID with group labels
        - metadata: dict with accession info and field used
    """
    gse = _parse_gse(accession, cache_dir)

    expression = _extract_expression(gse)
    if map_to_genes:
        expression = _map_probes_to_genes(expression, gse)

    expression = expression.apply(pd.to_numeric, errors="coerce")
    expression = expression.dropna(how="all")

    if group_column:
        labels: dict[str, str] = {}
        for gsm_name, gsm in gse.gsms.items():
            for key, values in gsm.metadata.items():
                if key.startswith("characteristics_ch"):
                    for v in values:
                        if ": " in v:
                            field, val = v.split(": ", 1)
                            if field.strip().lower() == group_column.lower():
                                labels[gsm_name] = val.strip()
        if not labels:
            raise ValueError(
                f"Group column '{group_column}' not found in sample characteristics. "
                f"Use list_sample_characteristics('{accession}') to see available fields."
            )
        field_used = group_column
    else:
        field_used, labels = _detect_group_field(gse)

    common_samples = [s for s in expression.columns if s in labels]
    if not common_samples:
        raise ValueError("No overlap between expression columns and samples with group labels.")

    expression = expression[common_samples]
    groups = pd.Series({s: labels[s] for s in common_samples})

    if control_label and treatment_label:
        label_map = {control_label: "control", treatment_label: "treatment"}
        groups = groups.map(lambda x: label_map.get(x, x))
        mask = groups.isin(["control", "treatment"])
        groups = groups[mask]
        expression = expression[groups.index]

    metadata = {
        "accession": accession,
        "title": gse.metadata.get("title", [""])[0] if hasattr(gse, "metadata") else "",
        "group_field": field_used,
        "group_values": sorted(groups.unique().tolist()),
        "n_samples": len(groups),
        "n_genes": len(expression),
        "platform": list(gse.gpls.keys()) if gse.gpls else [],
    }

    logger.info(
        "Loaded %s: %d genes x %d samples, groups: %s",
        accession,
        len(expression),
        len(groups),
        metadata["group_values"],
    )
    return expression, groups, metadata
