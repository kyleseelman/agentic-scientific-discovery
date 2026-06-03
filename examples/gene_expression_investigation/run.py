#!/usr/bin/env python3
"""Run an autonomous gene-expression investigation on the synthetic cohort."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EX_DIR = Path(__file__).resolve().parent
for p in (ROOT, EX_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

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

from synthetic_cohort import build_synthetic_cohort  # noqa: E402


def _print_gpu_info() -> None:
    """Print GPU availability and configuration."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            vram = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
            print(f"  VRAM: {vram / 1e9:.1f} GB")
            print(f"  PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
        else:
            print("GPU: Not available (running on CPU)")
    except ImportError:
        print("GPU: PyTorch not installed (running on CPU)")


def main() -> None:
    cfg = get_config()
    print("=" * 60)
    print("Agentic Scientific Discovery — GPU-Accelerated Run")
    print("=" * 60)
    _print_gpu_info()
    print(f"LLM provider: {cfg.llm_provider.value}")
    print(f"Device: {cfg.device}")
    print(f"GPU embeddings: {cfg.use_gpu_embeddings}")
    print(f"GPU compute: {cfg.use_gpu_compute}")
    print("=" * 60)

    session_dir = Path(__file__).resolve().parent / "session_demo"
    session_dir.mkdir(parents=True, exist_ok=True)
    knowledge_dir = session_dir / "knowledge"
    log_path = session_dir / "experiments.json"
    papers_dir = session_dir / "papers"

    expr, groups, pathway_sets = build_synthetic_cohort()
    output_dir = session_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

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
            "note": "Synthetic cohort with planted PLANTED_STRESS upregulation in treatment.",
            "n_pathways_defined": len(pathway_sets),
        }

    llm = create_llm_backend(cfg)
    knowledge = KnowledgeStore(knowledge_dir)
    experiments = ExperimentLog(log_path)
    paper_store = PaperStore(papers_dir)

    orch_cfg = OrchestratorConfig(
        research_question=(
            "What biological processes distinguish the treatment group from control?"
        ),
        available_data_keys=[
            "expression_matrix",
            "sample_groups",
            "pathway_gene_sets",
        ],
        budget=ResearchBudget(max_cycles=4, max_wall_time_s=600),
        strategy_every_n_cycles=2,
        literature_review_every_n_cycles=2,
        literature_enabled=True,
        max_papers_per_scan=3,
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

    cycles = orchestrator.run()
    orchestrator.save_session(session_dir)

    summary_out = session_dir / "run_summary.json"
    summary_out.write_text(json.dumps(cycles, indent=2, default=str))

    n_papers = len(paper_store.all_papers())
    n_findings = len(knowledge.all_findings())
    n_hypotheses = len(knowledge.all_hypotheses())

    gpu_de_count = sum(
        1
        for c in cycles
        for d in c.get("decisions", [])
        if d.get("step") == "execute"
        for t in d.get("trace", [])
        if t.get("result", {}).get("gpu_accelerated")
    )

    print(f"\nCompleted {len(cycles)} cycles.")
    print(f"  Papers read: {n_papers}")
    print(f"  Hypotheses generated: {n_hypotheses}")
    print(f"  Findings recorded: {n_findings}")
    print(f"  GPU-accelerated DE runs: {gpu_de_count}")
    print(f"  Device: {cfg.device}")
    print(f"  Artifacts under: {session_dir}")


if __name__ == "__main__":
    main()
