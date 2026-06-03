"""Knowledge graph query and write tools for the agentic research loop.

Exposes the bio-knowledge-graph-rag ``HybridRetriever`` and
``SessionIntegrator`` as agent tools registered in the shared
``TOOL_REGISTRY``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from src.tools.data_analysis import ToolContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports from bio-knowledge-graph-rag (sibling project)
# ---------------------------------------------------------------------------

_KG_AVAILABLE = False


def _ensure_kg_imports() -> bool:
    """Try to make bio-knowledge-graph-rag importable.

    The sibling project lives at the same level in the portfolio directory.
    We add its parent to ``sys.path`` if needed so that ``from src.…``
    relative imports inside that package resolve correctly.
    """
    global _KG_AVAILABLE
    if _KG_AVAILABLE:
        return True

    try:
        from src.hybrid_retriever import HybridRetriever  # noqa: F401

        _KG_AVAILABLE = True
        return True
    except ImportError:
        pass

    bio_kg_root = (
        Path(__file__).resolve().parents[3]
        / "bio-knowledge-graph-rag"
    )
    if bio_kg_root.is_dir() and str(bio_kg_root) not in sys.path:
        sys.path.insert(0, str(bio_kg_root))

    try:
        from src.hybrid_retriever import HybridRetriever  # noqa: F401

        _KG_AVAILABLE = True
    except ImportError:
        logger.debug(
            "bio-knowledge-graph-rag not importable; KG tools will return "
            "graceful fallback responses."
        )
    return _KG_AVAILABLE


# ---------------------------------------------------------------------------
# Singleton-style KG handles (lazily initialized once per process)
# ---------------------------------------------------------------------------

_retriever_instance = None
_integrator_instance = None


def _get_retriever():
    global _retriever_instance
    if _retriever_instance is not None:
        return _retriever_instance

    if not _ensure_kg_imports():
        return None

    try:
        from src.graph_store import BioGraphStore
        from src.hybrid_retriever import HybridRetriever
        from src.vector_store import MultiModalVectorStore

        import networkx as nx

        store = BioGraphStore(nx.MultiGraph())
        vector_store = MultiModalVectorStore()
        vector_store.create_predefined_collections()

        _retriever_instance = HybridRetriever(
            store=store,
            vector_store=vector_store,
        )
    except Exception as exc:
        logger.warning("Could not initialize KG retriever: %s", exc)
        return None

    return _retriever_instance


def _get_integrator():
    global _integrator_instance
    if _integrator_instance is not None:
        return _integrator_instance

    retriever = _get_retriever()
    if retriever is None:
        return None

    try:
        from src.session_integrator import SessionIntegrator

        _integrator_instance = SessionIntegrator(
            store=retriever.store,
            vector_store=retriever.vector_store,
        )
    except Exception as exc:
        logger.warning("Could not initialize SessionIntegrator: %s", exc)
        return None

    return _integrator_instance


def set_retriever(retriever: Any) -> None:
    """Inject an already-constructed HybridRetriever (e.g. with a loaded graph)."""
    global _retriever_instance
    _retriever_instance = retriever


def set_integrator(integrator: Any) -> None:
    """Inject an already-constructed SessionIntegrator."""
    global _integrator_instance
    _integrator_instance = integrator


# ---------------------------------------------------------------------------
# Tool functions (follow the TOOL_REGISTRY signature)
# ---------------------------------------------------------------------------


def query_knowledge_graph(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Query the biomedical knowledge graph for structured context about entities.

    Parameters (via *params* dict):
        query : str – free-text research question
        entity_names : list[str] – optional CURIE IDs or gene names
        hops : int – graph expansion radius (default 2)
        top_k : int – vector results to retrieve (default 6)
        mode : str – "auto", "graph", "vector", or "hybrid"
    """
    query = str(params.get("query", ""))
    entity_names = list(params.get("entity_names", []))
    hops = int(params.get("hops", 2))
    top_k = int(params.get("top_k", 6))
    mode = str(params.get("mode", "auto"))

    retriever = _get_retriever()
    if retriever is None:
        return {
            "context": (
                "Knowledge graph is not available in this environment. "
                "Install bio-knowledge-graph-rag as a sibling project."
            ),
            "source_nodes": [],
            "source_edges": [],
            "query_type": "unavailable",
        }

    result = retriever.retrieve(
        query=query,
        entity_names=entity_names if entity_names else None,
        hops=hops,
        top_k=top_k,
        mode=mode,
    )

    return {
        "context": result.context_text,
        "source_nodes": result.source_nodes,
        "source_edges": [
            {"source": s, "target": t, "relation": r}
            for s, t, r in result.source_edges
        ],
        "provenance": result.provenance[:20],
        "query_type": result.query_type,
    }


def add_to_knowledge_graph(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Persist findings from the current session into the knowledge graph.

    Parameters (via *params* dict):
        findings : list[dict] – finding dicts with keys ``statement``,
            ``evidence_strength``, ``entities``, etc.
        session_dir : str – path to the session directory produced by the
            orchestrator (optional; when given, the full session is ingested).
    """
    integrator = _get_integrator()
    if integrator is None:
        return {
            "status": "unavailable",
            "message": (
                "Knowledge graph integrator is not available. "
                "Install bio-knowledge-graph-rag as a sibling project."
            ),
        }

    session_dir = params.get("session_dir")
    if session_dir:
        try:
            stats = integrator.ingest_session(Path(session_dir))
            return {"status": "ok", **stats}
        except FileNotFoundError as exc:
            return {"status": "error", "message": str(exc)}

    findings = params.get("findings", [])
    if not findings:
        return {"status": "noop", "message": "No findings or session_dir provided."}

    nodes_added = 0
    edges_added = 0

    try:
        from src.schema import EdgeType, KGEdge, KGNode, NodeType, Provenance
    except ImportError:
        _ensure_kg_imports()
        from src.schema import EdgeType, KGEdge, KGNode, NodeType, Provenance

    for finding in findings:
        fid = finding.get("id", f"finding_inline_{nodes_added}")
        statement = finding.get("statement", "")
        node = KGNode(
            id=f"finding:{fid}",
            node_type=NodeType.EXPERIMENT,
            name=statement[:120] if statement else fid,
            description=statement,
            sources=[
                Provenance(
                    source_type="agent_discovery",
                    evidence_tier="bronze",
                    extraction_method="automated_agent",
                )
            ],
            properties={
                "evidence_strength": finding.get("evidence_strength"),
                "artifact_type": "finding",
            },
        )
        integrator.store.merge_node(node)
        nodes_added += 1

        for entity in finding.get("entities", []):
            if entity in integrator.store.graph:
                edge = KGEdge(
                    source_id=f"finding:{fid}",
                    target_id=entity,
                    edge_type=EdgeType.FINDING_ABOUT,
                    provenance=[
                        Provenance(
                            source_type="agent_discovery",
                            evidence_tier="bronze",
                        )
                    ],
                )
                integrator.store.add_typed_edge(edge)
                edges_added += 1

    return {
        "status": "ok",
        "nodes_added": nodes_added,
        "edges_added": edges_added,
    }


# ---------------------------------------------------------------------------
# Registration in TOOL_REGISTRY
# ---------------------------------------------------------------------------

def register_knowledge_graph_tools(registry: dict) -> None:
    """Add KG tools to an existing tool registry dict."""
    registry["query_knowledge_graph"] = query_knowledge_graph
    registry["add_to_knowledge_graph"] = add_to_knowledge_graph
