"""MSigDB Gene Set Loader — GMT format parser and downloader.

Downloads and parses MSigDB gene set collections (GMT format) into
dict[str, list[str]] compatible with ToolContext.pathway_sets.
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "agentic-discovery" / "msigdb"

MSIGDB_BASE_URL = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs"

COLLECTION_FILES: dict[str, str] = {
    "hallmark": "h.all.v2024.1.Hs.symbols.gmt",
    "c1_positional": "c1.all.v2024.1.Hs.symbols.gmt",
    "c2_cp": "c2.cp.v2024.1.Hs.symbols.gmt",
    "c2_cp_kegg": "c2.cp.kegg_medicus.v2024.1.Hs.symbols.gmt",
    "c2_cp_reactome": "c2.cp.reactome.v2024.1.Hs.symbols.gmt",
    "c2_cp_biocarta": "c2.cp.biocarta.v2024.1.Hs.symbols.gmt",
    "c3_mir": "c3.mir.v2024.1.Hs.symbols.gmt",
    "c5_go_bp": "c5.go.bp.v2024.1.Hs.symbols.gmt",
    "c5_go_mf": "c5.go.mf.v2024.1.Hs.symbols.gmt",
    "c5_go_cc": "c5.go.cc.v2024.1.Hs.symbols.gmt",
    "c6_oncogenic": "c6.all.v2024.1.Hs.symbols.gmt",
    "c7_immunologic": "c7.all.v2024.1.Hs.symbols.gmt",
    "c8_cell_type": "c8.all.v2024.1.Hs.symbols.gmt",
}


def _get_cache_dir(cache_dir: Path | None = None) -> Path:
    d = cache_dir or DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_gmt(filepath: Path) -> dict[str, list[str]]:
    """Parse a GMT file into a pathway dictionary.

    GMT format: each line is TAB-separated with:
        SET_NAME <TAB> description <TAB> GENE1 <TAB> GENE2 <TAB> ...

    Returns
    -------
    dict mapping set names to lists of gene symbols.
    """
    pathway_sets: dict[str, list[str]] = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            set_name = parts[0]
            genes = [g.strip() for g in parts[2:] if g.strip()]
            pathway_sets[set_name] = genes

    return pathway_sets


def _download_gmt(filename: str, cache_dir: Path | None = None) -> Path:
    """Download a GMT file from MSigDB if not cached."""
    dest_dir = _get_cache_dir(cache_dir)
    filepath = dest_dir / filename

    if filepath.exists():
        logger.info("Using cached GMT: %s", filepath)
        return filepath

    url = f"{MSIGDB_BASE_URL}/{filename}"
    logger.info("Downloading MSigDB collection: %s", url)

    try:
        urllib.request.urlretrieve(url, filepath)
    except Exception as e:
        filepath.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download {url}: {e}. "
            "MSigDB may require registration for some collections. "
            "The Hallmark collection (h.all) is freely available."
        ) from e

    logger.info("Downloaded %s (%d bytes)", filepath.name, filepath.stat().st_size)
    return filepath


def load_msigdb_collection(
    collection: str = "hallmark",
    cache_dir: Path | None = None,
) -> dict[str, list[str]]:
    """Download and parse an MSigDB gene set collection.

    Parameters
    ----------
    collection : str
        Collection identifier. One of: hallmark, c2_cp, c2_cp_kegg,
        c2_cp_reactome, c5_go_bp, c5_go_mf, c5_go_cc, c6_oncogenic,
        c7_immunologic, c8_cell_type, etc.
    cache_dir : Path, optional
        Cache directory for downloaded GMT files.

    Returns
    -------
    dict[str, list[str]]
        Mapping of pathway names to gene symbol lists.

    Raises
    ------
    ValueError
        If the collection name is not recognized.
    """
    if collection not in COLLECTION_FILES:
        raise ValueError(
            f"Unknown collection '{collection}'. "
            f"Available: {sorted(COLLECTION_FILES.keys())}"
        )

    filename = COLLECTION_FILES[collection]
    filepath = _download_gmt(filename, cache_dir)
    pathway_sets = parse_gmt(filepath)

    logger.info(
        "Loaded collection '%s': %d gene sets, median size %d genes",
        collection,
        len(pathway_sets),
        sorted(len(v) for v in pathway_sets.values())[len(pathway_sets) // 2]
        if pathway_sets
        else 0,
    )
    return pathway_sets


def load_gmt_file(filepath: str | Path) -> dict[str, list[str]]:
    """Load a local GMT file directly.

    Parameters
    ----------
    filepath : str or Path
        Path to a GMT format file.

    Returns
    -------
    dict[str, list[str]]
        Mapping of pathway names to gene symbol lists.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"GMT file not found: {path}")
    return parse_gmt(path)


def filter_pathway_sets(
    pathway_sets: dict[str, list[str]],
    gene_universe: set[str] | list[str],
    min_size: int = 5,
    max_size: int = 500,
) -> dict[str, list[str]]:
    """Filter pathway sets to genes present in the expression matrix.

    Parameters
    ----------
    pathway_sets : dict[str, list[str]]
        Full pathway dictionary.
    gene_universe : set or list of str
        Genes present in the expression matrix.
    min_size : int
        Minimum number of overlapping genes to keep a set.
    max_size : int
        Maximum number of overlapping genes to keep a set.

    Returns
    -------
    dict[str, list[str]]
        Filtered pathway dictionary with only genes in the universe.
    """
    universe = set(gene_universe)
    filtered: dict[str, list[str]] = {}

    for name, genes in pathway_sets.items():
        overlap = [g for g in genes if g in universe]
        if min_size <= len(overlap) <= max_size:
            filtered[name] = overlap

    logger.info(
        "Filtered %d -> %d pathway sets (universe size: %d, min_size=%d, max_size=%d)",
        len(pathway_sets),
        len(filtered),
        len(universe),
        min_size,
        max_size,
    )
    return filtered


def available_collections() -> list[str]:
    """Return list of known MSigDB collection identifiers."""
    return sorted(COLLECTION_FILES.keys())
