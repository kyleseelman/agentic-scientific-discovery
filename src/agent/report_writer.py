"""Structured arXiv-style research report generator.

Produces a proper research paper with sections generated individually using
focused LLM prompts and real structured data (not a single dump-everything prompt).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import LLMBackend


@dataclass
class ResearchReport:
    title: str = ""
    abstract: str = ""
    introduction: str = ""
    related_work: str = ""
    methods: str = ""
    experiments: str = ""
    results: str = ""
    discussion: str = ""
    conclusions: str = ""
    references: str = ""
    comparison_table: str = ""
    ablation_table: str = ""
    figures_dir: str = ""


def format_comparison_table(
    novel_models: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    metrics: list[str] | None = None,
) -> str:
    """Generate a markdown table comparing all models across metrics."""
    if metrics is None:
        all_keys: set[str] = set()
        for m in novel_models + baselines:
            all_keys.update(m.get("metrics", {}).keys())
        metrics = sorted(all_keys & {"accuracy", "f1", "auc_roc", "precision", "recall",
                                      "cross_val_mean", "cross_val_std"})
        if not metrics:
            metrics = sorted(all_keys)

    header = "| Model | Source | " + " | ".join(m.replace("_", " ").title() for m in metrics) + " |"
    sep = "|" + "|".join(["---"] * (len(metrics) + 2)) + "|"
    rows = []

    all_entries = []
    for m in novel_models:
        all_entries.append((m.get("name", "Novel Model"), "This work", m.get("metrics", {})))
    for b in baselines:
        all_entries.append((b.get("name", "Baseline"), b.get("source", ""), b.get("metrics", {})))

    best_per_metric = {}
    for metric in metrics:
        vals = [(e[0], e[2].get(metric, -1)) for e in all_entries]
        if vals:
            best_name = max(vals, key=lambda x: x[1])[0]
            best_per_metric[metric] = best_name

    for name, source, mets in all_entries:
        vals = []
        for metric in metrics:
            v = mets.get(metric)
            if v is not None:
                s = f"{v:.4f}"
                if best_per_metric.get(metric) == name:
                    s = f"**{s}**"
                vals.append(s)
            else:
                vals.append("—")
        rows.append(f"| {name} | {source} | " + " | ".join(vals) + " |")

    return "\n".join([header, sep] + rows)


def format_ablation_table(ablations: list[dict[str, Any]]) -> str:
    """Generate a markdown table for ablation study results."""
    if not ablations:
        return ""

    all_params: set[str] = set()
    all_metrics: set[str] = set()
    for a in ablations:
        all_params.update(a.get("varied_params", {}).keys())
        all_metrics.update(a.get("metrics", {}).keys())

    params = sorted(all_params)
    metrics = sorted(all_metrics & {"accuracy", "f1", "auc_roc"})
    if not metrics:
        metrics = sorted(all_metrics)[:3]

    header = "| " + " | ".join(params + metrics) + " |"
    sep = "|" + "|".join(["---"] * (len(params) + len(metrics))) + "|"
    rows = []
    for a in ablations:
        p_vals = [str(a.get("varied_params", {}).get(p, "—")) for p in params]
        m_vals = [f"{a.get('metrics', {}).get(m, 0):.4f}" if m in a.get("metrics", {}) else "—"
                  for m in metrics]
        rows.append("| " + " | ".join(p_vals + m_vals) + " |")

    return "\n".join([header, sep] + rows)


def generate_figures(
    novel_models: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    training_history: list[dict[str, Any]] | None = None,
    output_dir: str | Path = "figures",
) -> list[str]:
    """Generate matplotlib figures for the report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []

    all_models = []
    for m in novel_models:
        all_models.append({"name": m.get("name", "Novel"), "metrics": m.get("metrics", {}), "novel": True})
    for b in baselines:
        all_models.append({"name": b.get("name", "Baseline"), "metrics": b.get("metrics", {}), "novel": False})

    if all_models:
        metrics_to_plot = ["accuracy", "f1", "auc_roc"]
        available = [m for m in metrics_to_plot
                     if any(m in entry["metrics"] for entry in all_models)]

        if available:
            fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
            if len(available) == 1:
                axes = [axes]

            for ax, metric in zip(axes, available):
                names = [e["name"] for e in all_models if metric in e["metrics"]]
                values = [e["metrics"][metric] for e in all_models if metric in e["metrics"]]
                colors = ["#2196F3" if e["novel"] else "#9E9E9E"
                          for e in all_models if metric in e["metrics"]]

                bars = ax.barh(range(len(names)), values, color=colors)
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names, fontsize=9)
                ax.set_xlabel(metric.replace("_", " ").title())
                ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}")
                ax.set_xlim(0, 1.05)
                for bar, val in zip(bars, values):
                    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                            f"{val:.3f}", va="center", fontsize=8)

            plt.tight_layout()
            path = str(out / "model_comparison.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            figures.append(path)

    if training_history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for hist in training_history:
            name = hist.get("name", "Model")
            train_loss = hist.get("train_losses", [])
            val_loss = hist.get("val_losses", [])
            if train_loss:
                epochs = range(1, len(train_loss) + 1)
                ax1.plot(epochs, train_loss, label=f"{name} (train)", linestyle="-")
                if val_loss:
                    ax1.plot(epochs, val_loss, label=f"{name} (val)", linestyle="--")

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Curves")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        for hist in training_history:
            name = hist.get("name", "Model")
            val_acc = hist.get("val_accuracies", [])
            if val_acc:
                ax2.plot(range(1, len(val_acc) + 1), val_acc, label=name, marker="o", markersize=3)

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation Accuracy")
        ax2.set_title("Validation Accuracy over Training")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = str(out / "training_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures.append(path)

    return figures


def generate_report(
    research_context: dict[str, Any],
    novel_models: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    ablations: list[dict[str, Any]],
    llm: LLMBackend,
    training_history: list[dict[str, Any]] | None = None,
    output_dir: str | Path = ".",
    papers: list[dict[str, Any]] | None = None,
) -> ResearchReport:
    """Generate a complete arXiv-style research report section by section."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures_dir = out / "figures"

    comp_table = format_comparison_table(novel_models, baselines)
    abl_table = format_ablation_table(ablations)
    figure_paths = generate_figures(novel_models, baselines, training_history, figures_dir)

    task = research_context.get("task", "")
    dataset = research_context.get("dataset", "")
    dataset_desc = research_context.get("dataset_description", "")
    n_samples = research_context.get("n_samples", "?")
    n_genes = research_context.get("n_genes", "?")
    novel_arch = research_context.get("architecture", "")
    lit_summary = research_context.get("literature_summary", "")

    best_novel = {}
    if novel_models:
        best_novel = max(novel_models, key=lambda m: m.get("metrics", {}).get("accuracy", 0))
    best_baseline = {}
    if baselines:
        best_baseline = max(baselines, key=lambda b: b.get("metrics", {}).get("accuracy", 0))

    report = ResearchReport(figures_dir=str(figures_dir))

    # --- Title ---
    report.title = llm.generate(
        f"Generate a concise, specific research paper title for: {task} on {dataset} "
        f"using {novel_arch}. The key result is novel model vs baselines.",
        system="Return only the title, no quotes or extra text.", temperature=0.3,
    ).strip().strip('"')

    # --- Abstract ---
    report.abstract = llm.generate(
        f"""Write a 150-word scientific abstract for a paper about:
Task: {task}
Dataset: {dataset} ({n_samples} samples, {n_genes} genes)
Novel approach: {novel_arch}
Best novel model result: {json.dumps(best_novel.get('metrics', {}), default=str)}
Best baseline result: {best_baseline.get('name', 'N/A')} — {json.dumps(best_baseline.get('metrics', {}), default=str)}
Key finding: {'Novel model outperforms baselines' if best_novel.get('metrics', {}).get('accuracy', 0) > best_baseline.get('metrics', {}).get('accuracy', 0) else 'Baselines remain competitive'}""",
        system="Write a concise scientific abstract. No section headers.", temperature=0.3,
    )

    # --- Introduction ---
    report.introduction = llm.generate(
        f"""Write the Introduction section (~300 words) for a paper about:
Task: {task}
Dataset context: {dataset_desc[:500]}
Motivation: Why is this task important? What gap does this work address?
Contribution: We propose a {novel_arch} approach and compare against {len(baselines)} baselines.""",
        system="Scientific writing style. Cite relevant concepts, not specific papers.", temperature=0.3,
    )

    # --- Related Work ---
    papers_text = ""
    if papers:
        papers_text = "\n".join(f"- {p.get('title', '')} ({p.get('paper_id', '')})" for p in papers[:10])
    report.related_work = llm.generate(
        f"""Write the Related Work section (~250 words).
Task: {task}
Literature summary: {lit_summary[:1000]}
Key papers found:
{papers_text}
Cover: (1) prior work on this specific task/dataset, (2) relevant ML methods for biological data, (3) how our approach differs.""",
        system="Scientific writing. Reference concepts and methods. Be specific.", temperature=0.3,
    )

    # --- Methods ---
    report.methods = llm.generate(
        f"""Write the Methods section (~400 words).
Dataset: {dataset} — {dataset_desc[:300]}
Preprocessing: Top {n_genes} genes by variance, standardized
Novel architecture: {novel_arch}
Architecture details: {json.dumps(best_novel.get('architecture_config', {}), default=str)[:500]}
Baselines: {', '.join(b.get('name', '') for b in baselines)}
Training: Adam optimizer, early stopping, 80/10/10 train/val/test split
Evaluation: accuracy, F1, AUC-ROC, 5-fold cross-validation, statistical significance tests

Subsections: 2.1 Dataset, 2.2 Proposed Architecture, 2.3 Baselines, 2.4 Training, 2.5 Evaluation""",
        system="Technical methods description. Be precise about architecture and training details.", temperature=0.3,
    )

    # --- Experiments & Results ---
    report.results = f"""## Results

### Model Comparison

{comp_table}
"""
    if abl_table:
        report.results += f"""
### Ablation Study

{abl_table}
"""
    if figure_paths:
        for fp in figure_paths:
            fname = Path(fp).name
            report.results += f"\n![{fname}](figures/{fname})\n"

    results_narrative = llm.generate(
        f"""Write a Results narrative (~300 words) interpreting this comparison table:

{comp_table}

{'Ablation results: ' + abl_table if abl_table else ''}

Key questions to address:
1. Which model performed best and by how much?
2. Where did the novel model excel vs struggle?
3. Are the improvements statistically significant?
4. What do the ablation results show about design choices?""",
        system="Scientific results interpretation. Be precise with numbers.", temperature=0.3,
    )
    report.results += f"\n### Analysis\n\n{results_narrative}"

    # --- Discussion ---
    report.discussion = llm.generate(
        f"""Write the Discussion section (~300 words).
Task: {task}
Novel model: {novel_arch}
Best accuracy: {best_novel.get('metrics', {}).get('accuracy', 'N/A')}
Best baseline: {best_baseline.get('name', 'N/A')} at {best_baseline.get('metrics', {}).get('accuracy', 'N/A')}
{'Novel model outperforms' if best_novel.get('metrics', {}).get('accuracy', 0) > best_baseline.get('metrics', {}).get('accuracy', 0) else 'Baselines remain competitive'}

Discuss: (1) significance of results, (2) why the architecture works/doesn't, (3) limitations, (4) future work.""",
        system="Balanced scientific discussion. Acknowledge limitations honestly.", temperature=0.3,
    )

    # --- Conclusions ---
    report.conclusions = llm.generate(
        f"""Write a brief Conclusions section (~100 words).
Summarize: task, approach ({novel_arch}), key result ({json.dumps(best_novel.get('metrics', {}), default=str)}), 
and main takeaway for the {task} research community.""",
        system="Concise conclusions. No new information.", temperature=0.3,
    )

    # --- References ---
    refs = []
    if papers:
        for i, p in enumerate(papers[:15], 1):
            authors = p.get("authors", "Unknown")
            if isinstance(authors, list):
                authors = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            refs.append(f"[{i}] {authors}. \"{p.get('title', '')}\". {p.get('journal', '')}. {p.get('date', '')}.")
    report.references = "\n".join(refs) if refs else "No references collected."

    report.comparison_table = comp_table
    report.ablation_table = abl_table

    # --- Assemble full paper ---
    full_paper = f"""# {report.title}

## Abstract

{report.abstract}

## 1. Introduction

{report.introduction}

## 2. Related Work

{report.related_work}

## 3. Methods

{report.methods}

## 4. Experiments and Results

{report.results}

## 5. Discussion

{report.discussion}

## 6. Conclusions

{report.conclusions}

## References

{report.references}
"""
    (out / "research_paper.md").write_text(full_paper)

    return report
