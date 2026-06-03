#!/usr/bin/env python3
"""Run a novel research investigation from the command line.

This is the main entry point for running autonomous research. There are
three modes:

**Feed mode (--feed)**: Ingest the agent_feed.json from the
bio-literature-scanner. The agent reviews scored papers, selects
the most promising one, and runs a full investigation autonomously.

**Autonomous mode (--topic)**: Give a broad topic and the agent reads
papers, identifies gaps, formulates its own research questions, finds
a GEO dataset, and runs the full investigation — no human input needed.

**Directed mode (--question + --geo)**: Provide a specific question and
dataset for more controlled investigations.

Usage:
    # FEED: Consume daily literature scanner output
    python run_research.py \
        --feed ../bio-literature-scanner/agent_feed.json \
        --provider ollama --model qwen2.5:7b

    # FEED: Only investigate papers above score 8.0
    python run_research.py \
        --feed ../bio-literature-scanner/agent_feed.json \
        --feed-min-score 8.0 --provider openai --model gpt-4o-mini

    # AUTONOMOUS: Just give a topic — the agent does everything
    python run_research.py \
        --topic "Alzheimer's disease" \
        --provider ollama --model qwen2.5:7b

    # DIRECTED: Specify question + dataset
    python run_research.py \
        --question "What are the shared mechanisms between diabetes and Alzheimer's?" \
        --geo GSE5281 --group-column "disease state" \
        --control normal --treatment "Alzheimer's Disease" \
        --provider openai --model gpt-4o-mini

    # Demo mode (no API keys needed)
    python run_research.py --demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from src.agent.orchestrator import (
    OrchestratorConfig,
    ResearchBudget,
    ResearchOrchestrator,
)
from src.agent.question_discovery import discover_research_questions
from src.config import create_llm_backend, get_config
from src.data.geo_loader import load_geo_dataset, list_sample_characteristics
from src.data.msigdb_loader import load_msigdb_collection
from src.memory.experiment_log import ExperimentLog
from src.memory.knowledge_store import KnowledgeStore
from src.memory.paper_store import PaperStore
from src.tools.data_analysis import ToolContext


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autonomous scientific research agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--feed", "-f", type=str,
                    help="Path to agent_feed.json from bio-literature-scanner (feed mode)")
    p.add_argument("--feed-min-score", type=float, default=7.0,
                    help="Minimum overall score to consider a paper from the feed (default: 7.0)")
    p.add_argument("--feed-topic", type=str, default=None,
                    help="Only investigate papers from this topic in the feed")
    p.add_argument("--topic", "-t", type=str,
                    help="Broad topic — agent reads papers, finds gaps, generates questions (autonomous mode)")
    p.add_argument("--question", "-q", type=str,
                    help="Specific research question (directed mode)")
    p.add_argument("--geo", "-g", type=str,
                    help="GEO accession (e.g. GSE5281), required for directed mode")
    p.add_argument("--group-column", type=str, default=None,
                    help="Metadata field for sample grouping (auto-detected if omitted)")
    p.add_argument("--control", type=str, default=None,
                    help="Control group label")
    p.add_argument("--treatment", type=str, default=None,
                    help="Treatment group label")
    p.add_argument("--pathways", type=str, default="hallmark",
                    help="MSigDB collection (default: hallmark)")
    p.add_argument("--provider", type=str, default=None,
                    help="LLM provider: mock, ollama, openai, huggingface")
    p.add_argument("--model", type=str, default=None,
                    help="Model name for the chosen provider")
    p.add_argument("--cycles", type=int, default=3,
                    help="Max research cycles (default: 3)")
    p.add_argument("--output-dir", "-o", type=str, default="./research_output",
                    help="Output directory (default: ./research_output)")
    p.add_argument("--demo", action="store_true",
                    help="Run demo with mock LLM and synthetic data")
    p.add_argument("--organism", type=str, default="Homo sapiens")
    p.add_argument("--tissue", type=str, default=None)
    p.add_argument("--condition", type=str, default=None)
    return p.parse_args()


def run_demo(output_dir: Path) -> None:
    """Run a quick demo with mock LLM and synthetic data."""
    import numpy as np
    import pandas as pd

    print("=" * 70)
    print("  DEMO MODE — using mock LLM + synthetic data")
    print("=" * 70)

    os.environ["LLM_PROVIDER"] = "mock"
    cfg = get_config()
    llm = create_llm_backend(cfg)

    np.random.seed(42)
    n_genes, n_samples = 5000, 30
    genes = [f"GENE_{i}" for i in range(n_genes)]
    samples = [f"S{i}" for i in range(n_samples)]
    expression = pd.DataFrame(
        np.random.randn(n_genes, n_samples), index=genes, columns=samples
    )
    for i, g in enumerate(["TP53", "BRCA1", "MYC", "EGFR", "KRAS"]):
        expression.index.values[i] = g
        expression.iloc[i, n_samples // 2:] += 2.0
    groups = pd.Series(
        ["control"] * (n_samples // 2) + ["treatment"] * (n_samples // 2),
        index=samples,
    )

    run_pipeline(
        question="What genes distinguish treatment from control in this synthetic cohort?",
        expression=expression,
        groups=groups,
        metadata={"accession": "synthetic", "title": "Demo synthetic data"},
        pathways={},
        llm=llm,
        cfg=cfg,
        output_dir=output_dir,
        max_cycles=2,
        organism="Homo sapiens",
        tissue=None,
        condition=None,
    )


def run_pipeline(
    question: str,
    expression,
    groups,
    metadata: dict,
    pathways: dict,
    llm,
    cfg,
    output_dir: Path,
    max_cycles: int,
    organism: str,
    tissue: str | None,
    condition: str | None,
) -> None:
    start_time = time.time()

    session_dir = output_dir / "session"
    results_dir = output_dir / "results"
    session_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    knowledge = KnowledgeStore(session_dir / "knowledge")
    experiments = ExperimentLog(session_dir / "experiments.json")
    paper_store = PaperStore(session_dir / "papers")

    def make_ctx() -> ToolContext:
        return ToolContext(
            expression=expression, groups=groups,
            pathway_sets=pathways, output_dir=results_dir,
        )

    def get_summary() -> dict:
        g1, g2 = sorted(groups.unique())
        return {
            "accession": metadata.get("accession", "unknown"),
            "title": metadata.get("title", ""),
            "n_samples": int(expression.shape[1]),
            "n_genes": int(expression.shape[0]),
            "groups": {str(g1): int((groups == g1).sum()),
                       str(g2): int((groups == g2).sum())},
            "organism": organism,
            "tissue": tissue or "unknown",
            "condition": condition or "unknown",
        }

    orch_cfg = OrchestratorConfig(
        research_question=question,
        available_data_keys=["expression_matrix", "sample_groups", "pathway_gene_sets"],
        budget=ResearchBudget(max_cycles=max_cycles, max_wall_time_s=3600),
        strategy_every_n_cycles=2,
        literature_review_every_n_cycles=1,
        literature_enabled=True,
        max_papers_per_scan=3,
        organism=organism,
        tissue=tissue,
        condition=condition,
        auto_checkpoint=True,
        checkpoint_dir=str(session_dir),
    )

    orchestrator = ResearchOrchestrator(
        llm=llm, knowledge=knowledge, experiments=experiments,
        tool_ctx_factory=make_ctx, orch_cfg=orch_cfg,
        dataset_summary_provider=get_summary, config=cfg,
        paper_store=paper_store,
    )

    print(f"\n  Research question: {question[:90]}...")
    print(f"  Dataset: {metadata.get('accession', '?')} — {expression.shape[0]} genes × {expression.shape[1]} samples")
    print(f"  Groups: {dict(groups.value_counts())}")
    print(f"  LLM: {cfg.llm_provider.value}")
    print(f"  Max cycles: {max_cycles}")
    print()

    outcomes = orchestrator.run()
    orchestrator.save_session(session_dir)

    all_hypotheses = knowledge.all_hypotheses()
    all_findings = knowledge.all_findings()
    all_experiments = experiments.all()

    print(f"\n  Hypotheses: {len(all_hypotheses)}")
    print(f"  Findings: {len(all_findings)}")
    print(f"  Experiments: {len(all_experiments)}")

    # Generate report
    hyp_text = "\n".join(
        f"- [{h.status}, conf={h.confidence_prior:.2f}] {h.statement}"
        for h in all_hypotheses
    )
    find_text = "\n".join(
        f"- [strength={f.evidence_strength:.2f}] {f.statement}"
        for f in all_findings
    )
    exp_text = "\n".join(
        f"- {e.id}: verdict={e.interpretation.get('verdict','?')}, "
        f"{e.interpretation.get('summary','')[:150]}"
        for e in all_experiments if e.interpretation
    )
    papers_text = "\n".join(
        f"- {rp.paper.title} (PMID:{rp.paper.paper_id})"
        for rp in paper_store.all_papers()[:10]
    )

    report_prompt = f"""Write a scientific research report based on this autonomous investigation.

RESEARCH QUESTION: {question}

DATASET: {metadata.get('accession','?')} — {metadata.get('title','')}
{expression.shape[0]} genes, {expression.shape[1]} samples. Groups: {dict(groups.value_counts())}

HYPOTHESES: {hyp_text}
FINDINGS: {find_text}
EXPERIMENTS: {exp_text}
PAPERS: {papers_text}

Format: Abstract, Introduction, Methods, Results, Discussion, Conclusions, References.
Be specific with gene names, p-values, pathway names. Acknowledge limitations."""

    report = llm.generate(
        report_prompt,
        system="Scientific writer. Be precise, cite results, acknowledge limitations.",
        temperature=0.4,
    )
    report_path = output_dir / "research_report.md"
    report_path.write_text(report)

    elapsed = time.time() - start_time
    summary = {
        "research_question": question,
        "dataset": metadata.get("accession", "unknown"),
        "cycles": len(outcomes),
        "hypotheses": [{"id": h.id, "statement": h.statement, "status": h.status,
                        "confidence": h.confidence_prior} for h in all_hypotheses],
        "findings": [{"id": f.id, "statement": f.statement,
                      "evidence_strength": f.evidence_strength} for f in all_findings],
        "elapsed_s": round(elapsed),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    for h in all_hypotheses:
        print(f"  [{h.status:>13}] {h.statement[:70]}...")
    print(f"\n  Report: {report_path}")
    print(f"  Results: {results_dir}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print(f"{'=' * 70}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        run_demo(output_dir)
        return

    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        provider = (args.provider or os.environ.get("LLM_PROVIDER", "mock")).lower()
        if provider == "ollama":
            os.environ["OLLAMA_MODEL"] = args.model
        elif provider == "openai":
            os.environ["OPENAI_MODEL"] = args.model
        elif provider == "huggingface":
            os.environ["HF_MODEL"] = args.model

    cfg = get_config()
    llm = create_llm_backend(cfg)

    # ---- Feed mode: ingest scanner output ----
    if args.feed:
        run_from_feed(args, llm, cfg, output_dir)
        return

    # ---- Autonomous mode: discover questions from literature ----
    if args.topic:
        run_autonomous(args, llm, cfg, output_dir)
        return

    # ---- Directed mode: user specifies question + dataset ----
    if not args.question:
        print("Error: provide --feed, --topic (autonomous), --question + --geo (directed), or --demo")
        sys.exit(1)
    if not args.geo:
        print("Error: --geo is required in directed mode (e.g. --geo GSE5281)")
        sys.exit(1)

    run_directed(args, llm, cfg, output_dir)


def run_from_feed(args, llm, cfg, output_dir: Path) -> None:
    """Feed mode: ingest bio-literature-scanner output, pick the best paper, investigate."""
    feed_path = Path(args.feed)
    if not feed_path.exists():
        print(f"Error: feed file not found: {feed_path}")
        sys.exit(1)

    feed = json.loads(feed_path.read_text())

    print("=" * 70)
    print("  LITERATURE FEED INGESTION")
    print("=" * 70)
    print(f"  Feed: {feed_path}")
    print(f"  Generated: {feed.get('generated_at', 'unknown')}")
    print(f"  Total papers in feed: {feed.get('total_papers', 0)}")
    print(f"  Min score filter: {args.feed_min_score}")
    if args.feed_topic:
        print(f"  Topic filter: {args.feed_topic}")
    print()

    # Collect candidate papers from all topics
    candidates: list[dict] = []
    topics_data = feed.get("topics", {})
    for topic_name, papers in topics_data.items():
        if args.feed_topic and args.feed_topic.lower() not in topic_name.lower():
            continue
        for paper in papers:
            score = paper.get("score", 0)
            if score >= args.feed_min_score:
                paper["_topic"] = topic_name
                candidates.append(paper)

    candidates.sort(key=lambda p: p.get("research_potential", 0), reverse=True)

    if not candidates:
        print(f"  No papers above score {args.feed_min_score} in the feed.")
        print("  Lower the threshold with --feed-min-score or run a new scan.")
        sys.exit(0)

    print(f"  {len(candidates)} papers above threshold:\n")
    for i, p in enumerate(candidates[:10], 1):
        rp = p.get("research_potential", 0)
        print(f"  [{i}] [{p.get('score', 0):.1f} / rp:{rp:.0f}] {p['title'][:75]}...")
        if p.get("suggested_question"):
            print(f"      Q: {p['suggested_question'][:90]}...")
        print()

    # Use LLM to pick the most promising paper and formulate a research plan
    selection_prompt = f"""You are a research agent reviewing today's literature feed.
Below are the top-scoring papers. Pick the ONE paper with the highest potential
for a novel computational biology investigation using public gene expression data.

Papers:
"""
    for i, p in enumerate(candidates[:8], 1):
        selection_prompt += f"""
{i}. [{p.get('score', 0):.1f}] {p['title']}
   Topic: {p.get('_topic', '?')}
   Abstract: {p.get('abstract', '')[:300]}...
   Suggested question: {p.get('suggested_question', 'none')}
"""

    selection_prompt += """
Return JSON:
{
  "selected_index": <1-based index>,
  "research_question": "Your refined research question based on this paper",
  "rationale": "Why this paper is the best candidate",
  "suggested_geo": "A GEO accession to search for (e.g. GSE12345), or empty string",
  "search_terms": "GEO search terms to find a relevant dataset"
}"""

    print("  Agent is reviewing papers and selecting the best candidate...\n")
    response = llm.generate(
        selection_prompt,
        system="You are a computational biology research agent. Return valid JSON.",
        temperature=0.3,
    )

    # Parse selection
    start = response.find("{")
    end = response.rfind("}")
    selected_idx = 0
    research_question = ""
    geo_hint = ""

    if start != -1 and end != -1:
        try:
            sel = json.loads(response[start:end + 1])
            selected_idx = int(sel.get("selected_index", 1)) - 1
            research_question = sel.get("research_question", "")
            geo_hint = sel.get("suggested_geo", "")
            rationale = sel.get("rationale", "")
            search_terms = sel.get("search_terms", "")
            print(f"  Selected: Paper #{selected_idx + 1}")
            print(f"  Rationale: {rationale[:120]}...")
            print(f"  Research question: {research_question[:120]}...")
            if geo_hint:
                print(f"  Suggested GEO: {geo_hint}")
        except (json.JSONDecodeError, ValueError):
            pass

    selected_idx = max(0, min(selected_idx, len(candidates) - 1))
    selected_paper = candidates[selected_idx]

    if not research_question:
        research_question = selected_paper.get("suggested_question", selected_paper["title"])

    # Try the suggested GEO accession, or search for one
    from src.agent.question_discovery import search_geo_datasets

    expression = groups = metadata = None
    geo_to_try: list[str] = []

    if geo_hint and geo_hint.startswith("GSE"):
        geo_to_try.append(geo_hint)

    # Also search GEO for relevant datasets
    search_q = search_terms if 'search_terms' in dir() and search_terms else research_question[:80]
    print(f"\n  Searching GEO for datasets related to the question...")
    try:
        geo_results = search_geo_datasets(search_q, max_results=5)
        for gr in geo_results:
            acc = gr.get("accession", "")
            if acc.startswith("GSE") and acc not in geo_to_try:
                geo_to_try.append(acc)
                if len(geo_to_try) >= 5:
                    break
        if geo_results:
            print(f"  Found {len(geo_results)} potential datasets: {[g.get('accession','') for g in geo_results[:5]]}")
    except Exception as e:
        print(f"  GEO search failed: {e}")

    for acc in geo_to_try:
        print(f"\n  Trying {acc}...")
        try:
            expression, groups, metadata = load_geo_dataset(
                acc,
                group_column=args.group_column,
                control_label=args.control,
                treatment_label=args.treatment,
            )
            print(f"  Loaded: {expression.shape[0]} genes x {expression.shape[1]} samples")
            print(f"  Groups: {dict(groups.value_counts())}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if expression is None:
        print("\n  Could not load any dataset for this paper.")
        print("  The agent's selection and question have been saved.")
        print("  You can run manually with:")
        print(f"    python run_research.py --question \"{research_question[:80]}...\" --geo <accession>")
        feed_selection = {
            "selected_paper": selected_paper,
            "research_question": research_question,
            "attempted_datasets": geo_to_try,
        }
        (output_dir / "feed_selection.json").write_text(json.dumps(feed_selection, indent=2))
        sys.exit(0)

    try:
        pathways = load_msigdb_collection(args.pathways)
        print(f"  Pathways: {len(pathways)} {args.pathways} sets")
    except Exception:
        pathways = {}

    print()
    run_pipeline(
        question=research_question,
        expression=expression,
        groups=groups,
        metadata=metadata,
        pathways=pathways,
        llm=llm,
        cfg=cfg,
        output_dir=output_dir,
        max_cycles=args.cycles,
        organism=args.organism,
        tissue=args.tissue,
        condition=args.condition,
    )


def run_autonomous(args, llm, cfg, output_dir: Path) -> None:
    """Autonomous mode: agent reads papers, discovers questions, finds data, runs research."""
    print("=" * 70)
    print("  AUTONOMOUS RESEARCH DISCOVERY")
    print("=" * 70)
    print(f"  Topic: {args.topic}")
    print(f"  LLM: {cfg.llm_provider.value}")
    print(f"  The agent will now read papers, find gaps, and formulate research questions...")
    print()

    proposals = discover_research_questions(
        topic=args.topic, llm=llm, config=cfg,
        max_papers=8, max_proposals=3,
    )

    if not proposals:
        print("\n  The agent could not generate research proposals for this topic.")
        print("  Try a more specific topic or use directed mode (--question + --geo).")
        sys.exit(1)

    # Display proposals
    print(f"\n  Agent generated {len(proposals)} research proposals:\n")
    for i, p in enumerate(proposals, 1):
        print(f"  [{i}] {p.question}")
        print(f"      Rationale: {p.rationale[:120]}...")
        print(f"      Dataset: {p.suggested_geo} — {p.geo_rationale[:80]}...")
        print(f"      Novelty: {p.novelty[:80]}...")
        print(f"      Papers: {', '.join(p.source_papers[:3])}")
        print()

    # Save all proposals
    proposals_data = [
        {"question": p.question, "rationale": p.rationale,
         "source_papers": p.source_papers, "suggested_geo": p.suggested_geo,
         "geo_rationale": p.geo_rationale, "novelty": p.novelty,
         "feasibility": p.feasibility}
        for p in proposals
    ]
    (output_dir / "research_proposals.json").write_text(
        json.dumps(proposals_data, indent=2)
    )

    # Try each proposal until one loads successfully
    expression = groups = metadata = None
    selected = None

    for idx, proposal in enumerate(proposals):
        geo_accession = proposal.suggested_geo
        if not geo_accession or not geo_accession.startswith("GSE"):
            print(f"  Proposal #{idx+1}: no valid GEO accession ('{geo_accession}'), skipping")
            continue

        print(f"  Trying proposal #{idx+1}: {proposal.question[:70]}...")
        print(f"  Loading {geo_accession}...")

        try:
            expression, groups, metadata = load_geo_dataset(
                geo_accession,
                group_column=args.group_column,
                control_label=args.control,
                treatment_label=args.treatment,
            )
            selected = proposal
            print(f"  Loaded: {expression.shape[0]} genes × {expression.shape[1]} samples")
            print(f"  Groups: {dict(groups.value_counts())}")
            break
        except Exception as e:
            print(f"  Failed to load {geo_accession}: {e}")
            # Try to show available fields to help debugging
            try:
                chars = list_sample_characteristics(geo_accession)
                usable = {f: v for f, v in chars.items() if 2 <= len(v) <= 10}
                if usable:
                    print(f"  Available group columns: {list(usable.keys())}")
            except Exception:
                pass
            print(f"  Trying next proposal...\n")
            continue

    if selected is None or expression is None:
        print("\n  None of the proposed datasets could be loaded automatically.")
        print("  The agent proposed these datasets:")
        for p in proposals:
            print(f"    {p.suggested_geo}: {p.question[:60]}...")
        print("\n  To proceed, pick one and specify the group column manually:")
        print(f"  python run_research.py --topic \"{args.topic}\" "
              f"--geo <accession> --group-column \"<field>\"")
        sys.exit(1)

    try:
        pathways = load_msigdb_collection(args.pathways)
        print(f"  Pathways: {len(pathways)} {args.pathways} sets")
    except Exception:
        pathways = {}

    run_pipeline(
        question=selected.question,
        expression=expression,
        groups=groups,
        metadata=metadata,
        pathways=pathways,
        llm=llm,
        cfg=cfg,
        output_dir=output_dir,
        max_cycles=args.cycles,
        organism=args.organism,
        tissue=args.tissue,
        condition=args.condition,
    )


def run_directed(args, llm, cfg, output_dir: Path) -> None:
    """Directed mode: user provides question + dataset."""
    print("=" * 70)
    print("  AGENTIC SCIENTIFIC DISCOVERY")
    print("=" * 70)
    print(f"  Loading GEO dataset {args.geo}...")

    try:
        expression, groups, metadata = load_geo_dataset(
            args.geo,
            group_column=args.group_column,
            control_label=args.control,
            treatment_label=args.treatment,
        )
    except Exception as e:
        print(f"  Failed to load {args.geo}: {e}")
        print("  Tip: run `python -c \"from src.data.geo_loader import list_sample_characteristics; "
              f"print(list_sample_characteristics('{args.geo}'))\"` to see available fields")
        sys.exit(1)

    print(f"  Loaded: {expression.shape[0]} genes × {expression.shape[1]} samples")
    print(f"  Groups: {dict(groups.value_counts())}")

    try:
        pathways = load_msigdb_collection(args.pathways)
        print(f"  Pathways: {len(pathways)} {args.pathways} sets")
    except Exception as e:
        print(f"  Pathway loading failed ({e}), continuing without")
        pathways = {}

    run_pipeline(
        question=args.question,
        expression=expression,
        groups=groups,
        metadata=metadata,
        pathways=pathways,
        llm=llm,
        cfg=cfg,
        output_dir=output_dir,
        max_cycles=args.cycles,
        organism=args.organism,
        tissue=args.tissue,
        condition=args.condition,
    )


if __name__ == "__main__":
    main()
