"""Real data loaders for agentic scientific discovery.

Provides loaders for GEO expression data, MSigDB gene sets,
and HuggingFace datasets — all outputting formats compatible
with ToolContext (expression DataFrame, groups Series, pathway_sets dict).
"""
from src.data.geo_loader import (
    list_sample_characteristics,
    load_geo_dataset,
)
from src.data.hf_loader import (
    load_expression_csv,
    load_hf_omics_dataset,
)
from src.data.msigdb_loader import (
    available_collections,
    filter_pathway_sets,
    load_gmt_file,
    load_msigdb_collection,
    parse_gmt,
)

__all__ = [
    "load_geo_dataset",
    "list_sample_characteristics",
    "load_msigdb_collection",
    "load_gmt_file",
    "parse_gmt",
    "filter_pathway_sets",
    "available_collections",
    "load_hf_omics_dataset",
    "load_expression_csv",
]
