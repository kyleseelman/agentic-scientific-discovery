"""SOTA baseline scanner — find published results, train standard baselines, search pretrained models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.config import AppConfig, LLMBackend, get_config
from src.tools.literature import (
    PaperRecord,
    fetch_pubmed_abstracts,
    search_biorxiv,
    search_pubmed,
)
from src.utils.json_extract import extract_json_object


@dataclass
class BaselineRecord:
    name: str
    source: str  # "trained", "literature", "huggingface"
    paper: str
    metrics: dict[str, float]
    dataset: str
    model_path: str = ""
    notes: str = ""


@dataclass
class SOTALandscape:
    task: str
    dataset: str
    baselines: list[BaselineRecord] = field(default_factory=list)
    best_reported: BaselineRecord | None = None
    literature_summary: str = ""
    papers_scanned: int = 0


def scan_sota_literature(
    task: str,
    dataset: str,
    llm: LLMBackend,
    config: AppConfig | None = None,
    max_papers: int = 10,
) -> SOTALandscape:
    """Search literature for published benchmarks on the same task/dataset."""
    cfg = config or get_config()
    landscape = SOTALandscape(task=task, dataset=dataset)

    queries = [
        f"{task} {dataset} benchmark",
        f"{task} machine learning classification",
        f"{dataset} gene expression classification deep learning",
    ]

    all_papers: list[PaperRecord] = []
    seen_ids: set[str] = set()

    for q in queries:
        try:
            result = search_pubmed(q, config=cfg, retmax=max_papers)
            pmids = result.get("ids", [])
            papers = fetch_pubmed_abstracts(pmids, config=cfg)
            for p in papers:
                if p.paper_id not in seen_ids:
                    seen_ids.add(p.paper_id)
                    all_papers.append(p)
            time.sleep(0.5)
        except Exception:
            continue

    try:
        preprints = search_biorxiv(f"{task} benchmark", config=cfg, max_results=5)
        for p in preprints:
            if p.paper_id not in seen_ids:
                seen_ids.add(p.paper_id)
                all_papers.append(p)
    except Exception:
        pass

    landscape.papers_scanned = len(all_papers)

    if not all_papers:
        return landscape

    abstracts_text = "\n\n".join(
        f"[{p.paper_id}] {p.title}\n{p.abstract[:500]}"
        for p in all_papers[:15]
    )

    prompt = f"""Extract benchmark results from these paper abstracts.
Task: {task}
Dataset: {dataset}

Papers:
{abstracts_text}

For each paper that reports quantitative results (accuracy, AUC, F1, etc.) on this task or a closely related task, extract:
- model_name: the model or method name
- paper_id: the paper ID from above
- paper_citation: "Author et al. (Year)" format
- metrics: dict of metric_name -> value (as floats, e.g. 0.85 not 85%)
- dataset_used: which dataset they used
- notes: brief note on methodology

Return JSON:
{{
  "baselines": [
    {{"model_name": "...", "paper_id": "...", "paper_citation": "...",
      "metrics": {{"accuracy": 0.85, "auc_roc": 0.91}},
      "dataset_used": "...", "notes": "..."}}
  ],
  "summary": "Brief summary of the SOTA landscape for this task"
}}

If no quantitative results are found, return {{"baselines": [], "summary": "No published benchmarks found"}}."""

    try:
        text = llm.generate(prompt, system="Extract benchmark data. Return valid JSON only.", temperature=0.2)
        obj = extract_json_object(text)
    except Exception:
        obj = {"baselines": [], "summary": "Failed to parse literature results"}

    for b in obj.get("baselines", []):
        metrics = b.get("metrics", {})
        clean_metrics = {}
        for k, v in metrics.items():
            try:
                val = float(v)
                if val > 1.0:
                    val = val / 100.0
                clean_metrics[k] = val
            except (ValueError, TypeError):
                continue

        if clean_metrics:
            landscape.baselines.append(BaselineRecord(
                name=str(b.get("model_name", "Unknown")),
                source="literature",
                paper=str(b.get("paper_citation", "")),
                metrics=clean_metrics,
                dataset=str(b.get("dataset_used", dataset)),
                notes=str(b.get("notes", "")),
            ))

    landscape.literature_summary = str(obj.get("summary", ""))

    if landscape.baselines:
        best = max(landscape.baselines,
                   key=lambda r: r.metrics.get("accuracy", r.metrics.get("auc_roc", 0)))
        landscape.best_reported = best

    return landscape


def collect_baselines(
    task: str,
    ctx: Any,
    dataset_name: str = "",
) -> list[BaselineRecord]:
    """Train standard ML baselines on the current dataset."""
    from src.tools.data_analysis import run_tool

    baselines: list[BaselineRecord] = []
    methods = [
        ("Logistic Regression", "logistic_regression"),
        ("Random Forest", "random_forest"),
        ("SVM", "svm"),
        ("Gradient Boosting", "gradient_boosting"),
    ]

    for display_name, model_type in methods:
        try:
            result = run_tool("train_classifier", ctx, {
                "model_type": model_type,
                "n_top_genes": 200,
                "test_fraction": 0.3,
                "cv_folds": 5,
            })
            metrics: dict[str, float] = {}
            for k in ("accuracy", "f1", "auc_roc", "cross_val_mean"):
                if k in result and isinstance(result[k], (int, float)):
                    metrics[k] = float(result[k])

            baselines.append(BaselineRecord(
                name=display_name,
                source="trained",
                paper="",
                metrics=metrics,
                dataset=dataset_name,
                model_path=result.get("model_path", ""),
                notes=f"Trained on top 200 genes by variance, 5-fold CV",
            ))
            print(f"    Baseline {display_name}: {metrics}")
        except Exception as e:
            print(f"    Baseline {display_name} failed: {e}")

    return baselines


def search_pretrained_models(task: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search HuggingFace for pre-trained models relevant to the task."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        models = api.list_models(
            search=task,
            sort="downloads",
            limit=max_results,
        )
        return [
            {"model_id": m.modelId, "downloads": m.downloads, "tags": m.tags[:5]}
            for m in models
        ]
    except Exception:
        return []
