"""Knowledge graph query and write tools for the agentic research loop.

Loads a persistent ``KnowledgeEngine`` from the bio-knowledge-graph-rag
sibling project.  If no DB exists yet it auto-initializes with Hetionet
on first call.

Falls back to the legacy in-memory ``HybridRetriever`` when the engine
cannot be loaded, and degrades gracefully when neither is available.
"""

from __future__ import annotations

import logging
import os
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
    global _KG_AVAILABLE
    if _KG_AVAILABLE:
        return True

    bio_kg_root = (
        Path(__file__).resolve().parents[3]
        / "bio-knowledge-graph-rag"
    )
    if bio_kg_root.is_dir() and str(bio_kg_root) not in sys.path:
        sys.path.insert(0, str(bio_kg_root))

    try:
        from src.engine import KnowledgeEngine  # noqa: F401
        _KG_AVAILABLE = True
    except ImportError:
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
# Singleton handles
# ---------------------------------------------------------------------------

_engine_instance = None
_retriever_instance = None
_integrator_instance = None


def _get_engine():
    """Get or initialize the KnowledgeEngine (preferred persistent backend)."""
    global _engine_instance
    if _engine_instance is not None:
        return _engine_instance

    if not _ensure_kg_imports():
        return None

    try:
        from src.engine import KnowledgeEngine

        db_path = os.environ.get("KNOWLEDGE_DB",
                                 str(Path(__file__).resolve().parents[3]
                                     / "bio-knowledge-graph-rag" / "knowledge.db"))
        data_dir = os.environ.get("KNOWLEDGE_DATA_DIR",
                                  str(Path(__file__).resolve().parents[3]
                                      / "bio-knowledge-graph-rag" / "data"))

        _engine_instance = KnowledgeEngine(
            db_path=db_path, data_dir=data_dir,
        )

        stats = _engine_instance.graph.stats()
        if stats["total_nodes"] == 0:
            logger.info("KnowledgeEngine DB is empty — running init (Hetionet download)...")
            _engine_instance.init()

        logger.info("KnowledgeEngine loaded: %d nodes, %d edges",
                     stats["total_nodes"], stats["total_edges"])

    except Exception as exc:
        logger.warning("Could not initialize KnowledgeEngine: %s", exc)
        _engine_instance = None

    return _engine_instance


def _get_retriever():
    """Get HybridRetriever — tries KnowledgeEngine first, falls back to legacy."""
    global _retriever_instance
    if _retriever_instance is not None:
        return _retriever_instance

    engine = _get_engine()
    if engine is not None:
        return engine

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
        _retriever_instance = HybridRetriever(store=store, vector_store=vector_store)
    except Exception as exc:
        logger.warning("Could not initialize KG retriever: %s", exc)

    return _retriever_instance


def set_retriever(retriever: Any) -> None:
    """Inject an already-constructed HybridRetriever."""
    global _retriever_instance
    _retriever_instance = retriever


def set_integrator(integrator: Any) -> None:
    """Inject an already-constructed SessionIntegrator."""
    global _integrator_instance
    _integrator_instance = integrator


def set_engine(engine: Any) -> None:
    """Inject an already-constructed KnowledgeEngine."""
    global _engine_instance
    _engine_instance = engine


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def query_knowledge_graph(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Query the biomedical knowledge graph for structured context.

    Parameters (via *params* dict):
        query : str – free-text research question
        entity_names : list[str] – optional CURIE IDs or gene names
        hops : int – graph expansion radius (default 2)
        top_k : int – results to retrieve (default 10)
        mode : str – "auto", "graph", "vector", or "hybrid"
    """
    query = str(params.get("query", ""))
    entity_names = list(params.get("entity_names", []))
    hops = int(params.get("hops", 2))
    top_k = int(params.get("top_k", 10))
    mode = str(params.get("mode", "auto"))

    backend = _get_retriever()
    if backend is None:
        return {
            "context": "Knowledge graph is not available in this environment.",
            "source_nodes": [], "source_edges": [], "query_type": "unavailable",
        }

    engine = _get_engine()
    if engine is not None:
        result = engine.query(query, mode=mode, top_k=top_k)
        nodes_info = []
        edges_info = []
        for n in result.nodes[:top_k]:
            nodes_info.append({
                "id": n.id, "type": n.node_type.value,
                "name": n.name, "description": n.description[:300],
            })
        for e in result.edges[:20]:
            edges_info.append(e)

        if entity_names:
            for ename in entity_names:
                neighbor_result = engine.get_neighbors(ename, hops=hops)
                for n in neighbor_result.nodes:
                    if not any(ni["id"] == n.id for ni in nodes_info):
                        nodes_info.append({
                            "id": n.id, "type": n.node_type.value,
                            "name": n.name, "description": n.description[:300],
                        })
                edges_info.extend(neighbor_result.edges[:20])

        context_parts = [f"[{n['type']}] {n['name']}: {n['description']}"
                         for n in nodes_info[:15]]

        return {
            "context": "\n".join(context_parts) if context_parts else "No results found.",
            "source_nodes": [n["id"] for n in nodes_info],
            "source_edges": edges_info[:30],
            "query_type": result.mode,
            "num_results": len(nodes_info),
        }

    result = backend.retrieve(
        query=query, entity_names=entity_names or None,
        hops=hops, top_k=top_k, mode=mode,
    )
    return {
        "context": result.context_text,
        "source_nodes": result.source_nodes,
        "source_edges": [{"source": s, "target": t, "relation": r}
                         for s, t, r in result.source_edges],
        "provenance": result.provenance[:20],
        "query_type": result.query_type,
    }


def add_to_knowledge_graph(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Persist findings into the knowledge graph.

    Parameters (via *params* dict):
        findings : list[dict] – finding dicts
        session_dir : str – path to session directory (optional)
    """
    engine = _get_engine()

    if engine is not None:
        session_dir = params.get("session_dir")
        if session_dir:
            try:
                result = engine.ingest("agent_session", session_dir=session_dir)
                return {"status": "ok", **result}
            except Exception as exc:
                return {"status": "error", "message": str(exc)}

        findings = params.get("findings", [])
        if not findings:
            return {"status": "noop", "message": "No findings or session_dir provided."}

        from src.schema import KGNode, NodeType, Provenance

        nodes_added = 0
        for finding in findings:
            fid = finding.get("id", f"finding_inline_{nodes_added}")
            statement = finding.get("statement", "")
            node = KGNode(
                id=f"finding:{fid}",
                node_type=NodeType.EXPERIMENT,
                name=statement[:120] if statement else fid,
                description=statement,
                sources=[Provenance(source_type="agent_discovery",
                                    evidence_tier="bronze",
                                    extraction_method="automated_agent")],
                properties={"evidence_strength": finding.get("evidence_strength"),
                             "artifact_type": "finding",
                             "portfolio_type": "research_artifact"},
            )
            engine.graph.merge_node(node)
            nodes_added += 1

        engine.save_vectors()
        return {"status": "ok", "nodes_added": nodes_added}

    return {
        "status": "unavailable",
        "message": "Knowledge graph integrator is not available.",
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_knowledge_graph_tools(registry: dict) -> None:
    """Add KG tools to an existing tool registry dict."""
    registry["query_knowledge_graph"] = query_knowledge_graph
    registry["add_to_knowledge_graph"] = add_to_knowledge_graph
