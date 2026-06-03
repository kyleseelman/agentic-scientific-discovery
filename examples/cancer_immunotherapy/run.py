#!/usr/bin/env python3
"""Autonomous investigation of melanoma immunotherapy response using real GEO data.

Uses GSE91061 (Riaz et al. 2017) to investigate gene expression signatures
distinguishing anti-PD1 responders from non-responders in melanoma.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EX_DIR = Path(__file__).resolve().parent
for p in (ROOT, EX_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("MPLBACKEND", "Agg")

from src.agent.orchestrator import (  # noqa: E402
    OrchestratorConfig,
    ResearchBudget,
    ResearchOrchestrator,
)
from src.config import create_llm_backend, get_config  # noqa: E402
from src.memory.experiment_log import ExperimentLog  # noqa: E402
from src.memory.knowledge_store import KnowledgeStore  # noqa: E402
from src.memory.paper_store import PaperStore  # noqa: E402
from src.tools.data_analysis import ToolContext, profile_dataset  # noqa: E402
from src.data import load_geo_dataset, load_msigdb_collection, filter_pathway_sets  # noqa: E402


GEO_ACCESSION = "GSE91061"
GROUP_COLUMN = "response"
CONTROL_LABEL = "Non-responder"
TREATMENT_LABEL = "Responder"

ALTERNATIVE_ACCESSIONS = ["GSE78220", "GSE93157"]


def _load_expression_data() -> tuple:
    """Load GEO immunotherapy response data with fallback accessions."""
    accessions = [GEO_ACCESSION] + ALTERNATIVE_ACCESSIONS
    last_error = None

    for accession in accessions:
        try:
            print(f"  Trying {accession}...")
            if accession == GEO_ACCESSION:
                expr, groups, meta = load_geo_dataset(
                    accession,
                    group_column=GROUP_COLUMN,
                    control_label=CONTROL_LABEL,
                    treatment_label=TREATMENT_LABEL,
                )
            else:
                expr, groups, meta = load_geo_dataset(accession)
            print(f"  Successfully loaded {accession}: {len(expr)} genes x {len(groups)} samples")
            print(f"  Groups: {dict(groups.value_counts())}")
            return expr, groups, meta
        except Exception as e:
            last_error = e
            print(f"  Failed to load {accession}: {e}")
            continue

    raise RuntimeError(
        f"Could not load any immunotherapy dataset. Last error: {last_error}\n"
        "Ensure GEOparse is installed: pip install GEOparse\n"
        "And that you have network access to NCBI GEO."
    )


def main() -> None:
    cfg = get_config()
    print("=" * 60)
    print("Agentic Scientific Discovery — Cancer Immunotherapy Response")
    print("=" * 60)
    print(f"LLM provider: {cfg.llm_provider.value}")
    print(f"Device: {cfg.device}")
    print("=" * 60)

    session_dir = EX_DIR / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    output_dir = session_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    knowledge_dir = session_dir / "knowledge"
    log_path = session_dir / "experiments.json"
    papers_dir = session_dir / "papers"

    # --- Load real data ---
    print("\n[1/3] Loading GEO dataset...")
    expr, groups, geo_meta = _load_expression_data()

    print("\n[2/3] Loading MSigDB Hallmark gene sets...")
    pathway_sets = load_msigdb_collection("hallmark")
    pathway_sets = filter_pathway_sets(pathway_sets, set(expr.index))
    print(f"  Retained {len(pathway_sets)} pathway sets overlapping expression data")

    print("\n[3/3] Configuring orchestrator...")

    # --- Context factory ---
    def ctx_factory() -> ToolContext:
        return ToolContext(
            expression=expr.copy(),
            groups=groups.copy(),
            pathway_sets=pathway_sets,
            output_dir=output_dir,
        )

    def dataset_summary() -> dict:
        ctx = ctx_factory()
        prof = profile_dataset(ctx, {})
        return {
            **prof,
            **geo_meta,
            "n_pathways_defined": len(pathway_sets),
        }

    # --- Build orchestrator ---
    llm = create_llm_backend(cfg)
    knowledge = KnowledgeStore(knowledge_dir)
    experiments = ExperimentLog(log_path)
    paper_store = PaperStore(papers_dir)

    orch_cfg = OrchestratorConfig(
        research_question=(
            "What gene expression signatures distinguish immunotherapy "
            "responders from non-responders in melanoma?"
        ),
        available_data_keys=[
            "expression_matrix",
            "sample_groups",
            "pathway_gene_sets",
        ],
        budget=ResearchBudget(max_cycles=5, max_wall_time_s=900),
        strategy_every_n_cycles=2,
        literature_review_every_n_cycles=2,
        literature_enabled=True,
        max_papers_per_scan=3,
        organism="Homo sapiens",
        tissue="melanoma tumor",
        condition="anti-PD1 immunotherapy",
    )

    orchestrator = ResearchOrchestrator(
        llm=llm,
        knowledge=knowledge,
        experiments=experiments,
        tool_ctx_factory=ctx_factory,
        orch_cfg=orch_cfg,
        dataset_summary_provider=dataset_summary,
        config=cfg,
        paper_store=paper_store,
    )

    # --- Run investigation ---
    print("\nStarting autonomous investigation...\n")
    cycles = orchestrator.run()
    orchestrator.save_session(session_dir)

    # --- Save summary ---
    summary_out = session_dir / "run_summary.json"
    summary_out.write_text(json.dumps(cycles, indent=2, default=str))

    n_papers = len(paper_store.all_papers())
    n_findings = len(knowledge.all_findings())
    n_hypotheses = len(knowledge.all_hypotheses())

    print(f"\nCompleted {len(cycles)} research cycles.")
    print(f"  Papers read: {n_papers}")
    print(f"  Hypotheses generated: {n_hypotheses}")
    print(f"  Findings recorded: {n_findings}")
    print(f"  Artifacts under: {session_dir}")


if __name__ == "__main__":
    main()
