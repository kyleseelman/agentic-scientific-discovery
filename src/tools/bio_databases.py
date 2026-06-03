from __future__ import annotations

from typing import Any

import requests

from src.config import AppConfig, get_config
from src.tools.data_analysis import ToolContext


def string_protein_interactions(
    genes: list[str],
    species: int = 9606,
    required_score: int = 400,
    limit: int = 25,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = config or get_config()
    if not genes:
        return {"edges": [], "note": "no genes"}
    url = "https://string-db.org/api/json/interactions"
    params = {
        "identifiers": "\n".join(genes[:20]),
        "species": species,
        "required_score": required_score,
        "limit": limit,
    }
    try:
        r = requests.get(url, params=params, timeout=cfg.request_timeout_s)
        r.raise_for_status()
        data = r.json()
        edges = []
        for row in data:
            edges.append(
                {
                    "protein_a": row.get("preferredName_A"),
                    "protein_b": row.get("preferredName_B"),
                    "score": row.get("score"),
                }
            )
        return {"edges": edges, "n": len(edges)}
    except requests.RequestException as e:
        return {"edges": [], "error": str(e)}


def uniprot_gene_lookup(gene: str, config: AppConfig | None = None) -> dict[str, Any]:
    cfg = config or get_config()
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {"query": f"gene:{gene}+AND+reviewed:true", "format": "json", "size": 3}
    try:
        r = requests.get(url, params=params, timeout=cfg.request_timeout_s)
        r.raise_for_status()
        res = r.json().get("results", [])
        out = []
        for hit in res:
            acc = hit.get("primaryAccession")
            prot = hit.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {})
            if isinstance(prot, dict):
                title = prot.get("value", "")
            else:
                title = str(prot)
            out.append({"accession": acc, "protein_name": title})
        return {"gene": gene, "hits": out}
    except requests.RequestException as e:
        return {"gene": gene, "hits": [], "error": str(e)}


def quickgo_terms(gene: str, config: AppConfig | None = None) -> dict[str, Any]:
    cfg = config or get_config()
    url = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
    params = {"geneProductId": gene, "limit": 10}
    try:
        r = requests.get(url, params=params, timeout=cfg.request_timeout_s)
        if r.status_code != 200:
            return {"gene": gene, "terms": [], "status": r.status_code}
        js = r.json()
        terms = []
        for a in js.get("results", [])[:10]:
            terms.append(
                {
                    "go_id": a.get("goId"),
                    "aspect": a.get("aspect"),
                    "evidence": a.get("evidenceCode"),
                }
            )
        return {"gene": gene, "terms": terms}
    except requests.RequestException as e:
        return {"gene": gene, "terms": [], "error": str(e)}


def register_bio_tools(registry: dict) -> None:
    def string_network(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        genes = [str(g) for g in params.get("genes", [])]
        if not genes:
            genes = list(
                ctx.expression.var(axis=1).sort_values(ascending=False).head(12).index.astype(str)
            )
        return string_protein_interactions(genes)

    def uniprot_tool(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        return uniprot_gene_lookup(str(params.get("gene", "")))

    def quickgo_tool(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        return quickgo_terms(str(params.get("gene", "")))

    registry["string_network"] = string_network
    registry["uniprot_lookup"] = uniprot_tool
    registry["gene_ontology_quickgo"] = quickgo_tool
