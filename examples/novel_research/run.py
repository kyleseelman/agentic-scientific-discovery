"""Novel research: Shared molecular mechanisms between type 2 diabetes and Alzheimer's disease.

This is a growing area of research — epidemiological studies show T2D patients
have ~60% higher risk of Alzheimer's, but the shared molecular mechanisms
are poorly understood. This investigation uses gene expression data from
both conditions to identify common transcriptomic signatures.

Dataset: GSE5281 — Alzheimer's brain regions (entorhinal cortex, hippocampus, etc.)
161 samples (87 AD, 74 control), 21,655 genes

Usage:
    # With any LLM backend:
    export LLM_PROVIDER=ollama          # or openai, huggingface
    export OLLAMA_MODEL=qwen2.5:7b      # model for your chosen backend
    python examples/novel_research/run.py

    # With OpenAI:
    export LLM_PROVIDER=openai
    export OPENAI_API_KEY=sk-...
    python examples/novel_research/run.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MPLBACKEND", "Agg")

from src.agent.orchestrator import (
    OrchestratorConfig,
    ResearchBudget,
    ResearchOrchestrator,
)
from src.config import create_llm_backend, get_config
from src.data.geo_loader import load_geo_dataset
from src.data.msigdb_loader import load_msigdb_collection
from src.memory.experiment_log import ExperimentLog
from src.memory.knowledge_store import KnowledgeStore
from src.memory.paper_store import PaperStore
from src.tools.data_analysis import ToolContext

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESEARCH_QUESTION = (
    "What are the shared molecular mechanisms and gene expression signatures "
    "between type 2 diabetes and Alzheimer's disease in human brain tissue? "
    "Identify genes, pathways, and potential causal drivers that link "
    "metabolic dysfunction to neurodegeneration."
)

GEO_ACCESSION = "GSE5281"
GROUP_COLUMN = "disease state"
CONTROL_LABEL = "normal"
TREATMENT_LABEL = "Alzheimer's Disease"

SESSION_DIR = Path(__file__).parent / "session"
OUTPUT_DIR = Path(__file__).parent / "outputs"
REPORT_PATH = Path(__file__).parent / "research_report.md"

MAX_CYCLES = 3


def main() -> None:
    start_time = time.time()
    print("=" * 70)
    print("  NOVEL RESEARCH: Diabetes-Alzheimer's Molecular Link")
    print("=" * 70)
    print(f"  Research question: {RESEARCH_QUESTION[:100]}...")
    print(f"  Dataset: {GEO_ACCESSION}")
    print(f"  Max cycles: {MAX_CYCLES}")
    print()

    # ---- 1. Load real data ------------------------------------------------
    print("[1/6] Loading GEO dataset and pathway gene sets...")
    try:
        expression, groups, metadata = load_geo_dataset(
            GEO_ACCESSION,
            group_column=GROUP_COLUMN,
            control_label=CONTROL_LABEL,
            treatment_label=TREATMENT_LABEL,
        )
        print(f"  Expression matrix: {expression.shape[0]} genes × {expression.shape[1]} samples")
        print(f"  Groups: {dict(groups.value_counts())}")
    except Exception as e:
        print(f"  GEO load failed ({e}), generating synthetic data for demo")
        import numpy as np
        import pandas as pd
        np.random.seed(42)
        n_genes, n_samples = 8000, 40
        genes = [f"GENE_{i}" for i in range(n_genes)]
        samples = [f"S{i}" for i in range(n_samples)]
        base = np.random.randn(n_genes, n_samples)
        # Insert real signal: insulin signaling + neuroinflammation genes
        signal_genes = ["INS", "INSR", "IRS1", "IRS2", "PIK3CA", "AKT1", "MTOR",
                        "APP", "PSEN1", "MAPT", "TREM2", "APOE", "CLU", "BIN1",
                        "GSK3B", "IDE", "BACE1", "IL6", "TNF", "IL1B"]
        for i, g in enumerate(signal_genes):
            genes[i] = g
            base[i, n_samples//2:] += np.random.uniform(1.5, 3.0)
        expression = pd.DataFrame(base, index=genes, columns=samples)
        groups = pd.Series(
            [CONTROL_LABEL]*(n_samples//2) + [TREATMENT_LABEL]*(n_samples//2),
            index=samples,
        )
        metadata = {"accession": "synthetic", "title": "Synthetic AD vs Control"}

    # Load pathway gene sets
    try:
        pathways = load_msigdb_collection("hallmark")
        print(f"  Loaded {len(pathways)} Hallmark pathway sets")
    except Exception as e:
        print(f"  MSigDB load failed ({e}), using empty pathways")
        pathways = {}

    # ---- 2. Initialize infrastructure ------------------------------------
    print("\n[2/6] Initializing research infrastructure...")
    cfg = get_config()
    llm = create_llm_backend(cfg)
    print(f"  LLM provider: {cfg.llm_provider.value}")

    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    knowledge = KnowledgeStore(SESSION_DIR / "knowledge")
    experiments = ExperimentLog(SESSION_DIR / "experiments.json")
    paper_store = PaperStore(SESSION_DIR / "papers")

    def make_ctx() -> ToolContext:
        return ToolContext(
            expression=expression,
            groups=groups,
            pathway_sets=pathways,
            output_dir=OUTPUT_DIR,
        )

    def get_summary() -> dict:
        g1, g2 = sorted(groups.unique())
        return {
            "accession": metadata.get("accession", GEO_ACCESSION),
            "title": metadata.get("title", ""),
            "n_samples": int(expression.shape[1]),
            "n_genes": int(expression.shape[0]),
            "groups": {str(g1): int((groups == g1).sum()),
                       str(g2): int((groups == g2).sum())},
            "organism": "Homo sapiens",
            "tissue": "brain",
            "condition": "Alzheimer's disease vs control",
            "research_focus": "diabetes-Alzheimer's molecular link",
        }

    # ---- 3. Configure & run orchestrator ---------------------------------
    print("\n[3/6] Running research orchestrator...")
    orch_cfg = OrchestratorConfig(
        research_question=RESEARCH_QUESTION,
        available_data_keys=["expression_matrix", "sample_groups", "pathway_gene_sets"],
        budget=ResearchBudget(max_cycles=MAX_CYCLES, max_wall_time_s=1800),
        strategy_every_n_cycles=2,
        literature_review_every_n_cycles=1,
        literature_enabled=True,
        max_papers_per_scan=3,
        organism="Homo sapiens",
        tissue="brain",
        condition="Alzheimer's disease",
        auto_checkpoint=True,
        checkpoint_dir=str(SESSION_DIR),
    )

    orchestrator = ResearchOrchestrator(
        llm=llm,
        knowledge=knowledge,
        experiments=experiments,
        tool_ctx_factory=make_ctx,
        orch_cfg=orch_cfg,
        dataset_summary_provider=get_summary,
        config=cfg,
        paper_store=paper_store,
    )

    outcomes = orchestrator.run()
    orchestrator.save_session(SESSION_DIR)

    # ---- 4. Collect results -----------------------------------------------
    print(f"\n[4/6] Collecting results from {len(outcomes)} cycles...")

    all_hypotheses = knowledge.all_hypotheses()
    all_findings = knowledge.all_findings()
    all_experiments = experiments.all()

    print(f"  Hypotheses generated: {len(all_hypotheses)}")
    print(f"  Findings recorded: {len(all_findings)}")
    print(f"  Experiments run: {len(all_experiments)}")

    for h in all_hypotheses:
        print(f"    [{h.status}] {h.statement[:80]}...")

    # ---- 5. Generate research report --------------------------------------
    print("\n[5/6] Generating research report with LLM...")

    hypotheses_text = "\n".join(
        f"- [{h.status}, posterior={h.confidence_prior:.2f}] {h.statement}"
        for h in all_hypotheses
    )
    findings_text = "\n".join(
        f"- [strength={f.evidence_strength:.2f}] {f.statement}"
        for f in all_findings
    )

    experiment_summaries = []
    for exp in all_experiments:
        interp = exp.interpretation or {}
        experiment_summaries.append(
            f"Experiment {exp.id}: hypothesis={exp.hypothesis_id}, "
            f"verdict={interp.get('verdict','?')}, "
            f"summary={interp.get('summary','')[:200]}"
        )
    experiments_text = "\n".join(experiment_summaries)

    papers_text = ""
    if paper_store:
        for rp in paper_store.all_papers()[:10]:
            p = rp.paper
            papers_text += f"- {p.title} (PMID:{p.paper_id})\n"

    report_prompt = f"""Write a comprehensive scientific research report based on the following
autonomous research investigation.

RESEARCH QUESTION: {RESEARCH_QUESTION}

DATASET: {GEO_ACCESSION} — {metadata.get('title', 'Brain tissue expression')}
Samples: {expression.shape[1]}, Genes: {expression.shape[0]}
Groups: {dict(groups.value_counts())}

HYPOTHESES TESTED:
{hypotheses_text}

KEY FINDINGS:
{findings_text}

EXPERIMENT DETAILS:
{experiments_text}

PAPERS REFERENCED:
{papers_text}

Write the report in standard scientific format:
1. Abstract (150 words)
2. Introduction — why this question matters, current state of knowledge
3. Methods — dataset, tools used, statistical approaches
4. Results — for each hypothesis tested, describe what was found with specific numbers
5. Discussion — interpretation, how findings connect, limitations
6. Conclusions — key takeaways and future directions
7. References — cite papers found during literature review

Be specific with gene names, p-values, fold changes, and pathway names.
If data is limited, acknowledge it and suggest follow-up experiments.
"""

    report = llm.generate(
        report_prompt,
        system="You are a scientific writer producing a research report. "
               "Be precise, cite specific results, and acknowledge limitations.",
        temperature=0.4,
    )

    REPORT_PATH.write_text(report)
    print(f"  Report saved to {REPORT_PATH}")

    # ---- 6. Summary -------------------------------------------------------
    elapsed = time.time() - start_time
    print(f"\n[6/6] Research complete in {elapsed:.0f}s")
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    for h in all_hypotheses:
        print(f"  [{h.status:>13}] {h.statement[:70]}...")
    print(f"\n  Total findings: {len(all_findings)}")
    print(f"  Report: {REPORT_PATH}")
    print(f"  Session data: {SESSION_DIR}")
    print("=" * 70)

    # Save structured summary
    summary = {
        "research_question": RESEARCH_QUESTION,
        "dataset": GEO_ACCESSION,
        "cycles_completed": len(outcomes),
        "hypotheses": [
            {"id": h.id, "statement": h.statement, "status": h.status,
             "confidence": h.confidence_prior}
            for h in all_hypotheses
        ],
        "findings": [
            {"id": f.id, "statement": f.statement,
             "evidence_strength": f.evidence_strength}
            for f in all_findings
        ],
        "elapsed_s": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (Path(__file__).parent / "run_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
