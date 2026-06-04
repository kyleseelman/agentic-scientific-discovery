"""Fully autonomous model research agent.

An LLM-driven research agent that autonomously:
- Scans literature for SOTA baselines and relevant architectures
- Trains standard baselines for comparison
- Searches and evaluates HuggingFace pretrained models
- Designs novel architectures (from papers or LLM reasoning)
- Trains with k-fold cross-validation
- Analyzes results, diagnoses failures, and decides next steps
- Runs hyperparameter search (random + LLM-guided)
- Iterates until it beats baselines or exhausts budget
- Writes an arXiv-style research paper

The LLM acts as the "brain" — at each iteration it sees the full research
log (all attempts, training curves, errors) and decides what to try next.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agent.report_writer import generate_report
from src.agent.sota_scanner import (
    BaselineRecord,
    collect_baselines,
    scan_sota_literature,
    search_pretrained_models,
)
from src.config import AppConfig, LLMBackend, get_config
from src.tools.data_analysis import ToolContext, run_tool
from src.utils.json_extract import extract_json_object


# ─── Utilities ──────────────────────────────────────────────────────────

def _valid_heads(hidden_dim: int) -> int:
    for c in [8, 4, 2, 1]:
        if hidden_dim % c == 0:
            return c
    return 1


def _auto_size(n_samples: int) -> dict[str, Any]:
    """Auto-size model and training params for dataset."""
    n_train = int(n_samples * 0.7)
    if n_train < 100:
        hd, nl, do, bs = 32, 2, 0.3, 8
    elif n_train < 300:
        hd, nl, do, bs = 48, 2, 0.3, 16
    elif n_train < 1000:
        hd, nl, do, bs = 64, 3, 0.2, 32
    else:
        hd, nl, do, bs = 128, 3, 0.1, 32

    epochs = max(100, min(500, 50000 // max(n_train, 1)))
    patience = max(15, epochs // 5)
    wd = 5e-3 if n_train < 200 else 1e-3 if n_train < 500 else 1e-4

    return {
        "hidden_dim": hd, "n_layers": nl, "n_heads": _valid_heads(hd),
        "dropout": do, "batch_size": bs, "epochs": epochs,
        "patience": patience, "weight_decay": wd,
    }


# ─── Research Agent ─────────────────────────────────────────────────────

class AutonomousResearchAgent:
    """LLM-driven autonomous model research agent."""

    def __init__(
        self,
        llm: LLMBackend,
        config: AppConfig | None = None,
        output_dir: str | Path = "research_output",
        max_iterations: int = 8,
    ) -> None:
        self.llm = llm
        self.cfg = config or get_config()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_iterations = max_iterations

        self.research_log: list[dict[str, Any]] = []
        self.all_models: list[dict[str, Any]] = []
        self.best_model: dict[str, Any] | None = None
        self.best_baseline_acc = 0.0
        self.baselines: list[BaselineRecord] = []
        self.papers_found: list[dict[str, Any]] = []
        self.hf_models_tried: list[dict[str, Any]] = []

    def run(
        self,
        task: str,
        expression: pd.DataFrame,
        groups: pd.Series,
        metadata: dict[str, Any],
        architecture: str = "attention_gene_network",
        n_top_genes: int = 200,
        run_ablation: bool = True,
    ) -> dict[str, Any]:
        start = time.time()
        n_samples = expression.shape[1]
        n_classes = len(groups.unique())
        dataset = metadata.get("accession", "unknown")
        sizing = _auto_size(n_samples)

        self._log("init", f"Starting research: {task} on {dataset}", {
            "n_samples": n_samples, "n_genes": expression.shape[0],
            "n_classes": n_classes, "n_top_genes": n_top_genes,
            "auto_sizing": sizing,
        })

        print(f"\n{'='*70}")
        print(f"  Autonomous Research Agent")
        print(f"{'='*70}")
        print(f"  Task:      {task}")
        print(f"  Dataset:   {dataset} ({n_samples} samples)")
        print(f"  Budget:    {self.max_iterations} iterations")
        print(f"  Auto-size: hd={sizing['hidden_dim']}, "
              f"layers={sizing['n_layers']}, dropout={sizing['dropout']}")
        print(f"{'='*70}\n")

        ctx = ToolContext(
            expression=expression, groups=groups, output_dir=self.output_dir,
        )

        # ── Phase 1: Literature scan ────────────────────────────────
        print("[Phase 1] Literature & SOTA scan...")
        landscape = scan_sota_literature(task, dataset, self.llm, self.cfg)
        self.papers_found = [
            {"name": b.name, "paper": b.paper, "metrics": b.metrics}
            for b in landscape.baselines
        ]
        self._log("literature", f"Scanned {landscape.papers_scanned} papers", {
            "literature_baselines": len(landscape.baselines),
            "summary": landscape.literature_summary,
        })

        # ── Phase 2: Train baselines ────────────────────────────────
        print("\n[Phase 2] Training standard baselines...")
        self.baselines = collect_baselines(task, ctx, dataset)
        self.best_baseline_acc = max(
            (b.metrics.get("accuracy", 0) for b in self.baselines), default=0
        )
        baseline_summary = [
            {"name": b.name, "accuracy": b.metrics.get("accuracy", 0),
             "f1": b.metrics.get("f1", 0), "cv_mean": b.metrics.get("cross_val_mean", 0)}
            for b in self.baselines
        ]
        self._log("baselines", f"Best baseline: {self.best_baseline_acc:.4f}", {
            "baselines": baseline_summary,
        })
        print(f"  Best baseline: {self.best_baseline_acc:.4f}")

        # ── Phase 3: Search HuggingFace pretrained models ───────────
        print("\n[Phase 3] Searching HuggingFace for pretrained models...")
        hf_results = search_pretrained_models(task, max_results=5)
        if hf_results:
            self._log("hf_search", f"Found {len(hf_results)} HF models", {
                "models": hf_results,
            })
            print(f"  Found {len(hf_results)} relevant models: "
                  f"{[m['model_id'] for m in hf_results[:3]]}")
        else:
            self._log("hf_search", "No HF models found", {})

        # ── Phase 4: LLM-driven iterative model building ───────────
        print(f"\n[Phase 4] Autonomous model building "
              f"(up to {self.max_iterations} iterations)...")

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'─'*60}")
            print(f"  Iteration {iteration}/{self.max_iterations}")
            print(f"{'─'*60}")

            # Agent THINKS: analyze history and decide next action
            action = self._think(
                task, architecture, n_top_genes, n_classes,
                n_samples, sizing, iteration,
            )

            action_type = action.get("action", "train_model")
            print(f"  Agent decision: {action_type}")
            print(f"  Reasoning: {action.get('reasoning', '?')[:120]}")

            if action_type == "stop":
                print(f"  Agent decided to stop: {action.get('reasoning', '')}")
                self._log("stop", action.get("reasoning", "Agent decided to stop"), {})
                break

            # Execute the chosen action
            result = self._execute_action(
                action, ctx, task, n_top_genes, n_classes, sizing,
            )

            if result and "accuracy" in result:
                acc = result["accuracy"]
                gap = self.best_baseline_acc - acc
                print(f"  Result: accuracy={acc:.4f}, f1={result.get('f1', 0):.4f}")
                if gap > 0:
                    print(f"  Gap to baseline: {gap:.4f} ({gap/max(self.best_baseline_acc, 0.01):.1%})")
                else:
                    print(f"  BEATS best baseline by {-gap:.4f}!")

                if acc > (self.best_model or {}).get("metrics", {}).get("accuracy", 0):
                    self.best_model = {
                        "name": result.get("name", f"Model iter {iteration}"),
                        "source": "This work",
                        "metrics": {
                            k: result[k] for k in ("accuracy", "f1", "best_val_loss",
                                                     "cv_mean", "cv_std")
                            if k in result
                        },
                        "architecture_config": result.get("config", {}),
                        "model_path": result.get("model_path", ""),
                        "training_history": {
                            "name": result.get("name", "Novel"),
                            "train_losses": result.get("train_losses", []),
                            "val_losses": result.get("val_losses", []),
                        },
                        "iteration": iteration,
                        "description": action.get("reasoning", ""),
                    }

                if acc >= self.best_baseline_acc:
                    self._log("milestone", "Novel model beats baselines!", {
                        "accuracy": acc, "baseline": self.best_baseline_acc,
                    })

            elif result and "error" in result:
                print(f"  FAILED: {result['error'][:120]}")

        # ── Phase 5: Ablation studies ───────────────────────────────
        ablations = []
        if run_ablation and self.best_model:
            print(f"\n[Phase 5] Running ablation studies...")
            ablations = self._run_ablation_studies(
                ctx, n_classes, n_top_genes, sizing,
                expression.shape[0], n_samples,
            )

        # ── Phase 6: K-fold cross-validation of best model ─────────
        if self.best_model and self.best_model.get("model_path"):
            print(f"\n[Phase 6] K-fold cross-validation of best model...")
            cv_result = self._kfold_cv(ctx, self.best_model, n_top_genes, n_classes)
            if cv_result:
                self.best_model["metrics"]["cv_mean"] = cv_result["cv_mean"]
                self.best_model["metrics"]["cv_std"] = cv_result["cv_std"]
                self.best_model["metrics"]["cv_scores"] = cv_result["cv_scores"]
                print(f"  CV: {cv_result['cv_mean']:.4f} ± {cv_result['cv_std']:.4f}")

        # ── Phase 7: Generate report ────────────────────────────────
        print(f"\n[Phase 7] Generating arXiv-style research paper...")

        all_baselines_dicts = []
        for b in landscape.baselines:
            all_baselines_dicts.append(asdict(b))
        for b in self.baselines:
            all_baselines_dicts.append(asdict(b))

        novel_models = [self.best_model] if self.best_model else []
        training_histories = [
            m.get("training_history") for m in novel_models
            if m.get("training_history")
        ]

        report = generate_report(
            research_context={
                "task": task, "dataset": dataset,
                "dataset_description": metadata.get("title", ""),
                "n_samples": n_samples, "n_genes": n_top_genes,
                "architecture": self.best_model.get("name", architecture) if self.best_model else architecture,
                "literature_summary": landscape.literature_summary,
                "iterations": len(self.all_models),
            },
            novel_models=novel_models,
            baselines=all_baselines_dicts,
            ablations=ablations,
            llm=self.llm,
            training_history=training_histories,
            output_dir=self.output_dir,
            papers=[{"title": p["name"], "authors": p["paper"]}
                    for p in self.papers_found],
        )

        elapsed = time.time() - start
        best_acc = self.best_model["metrics"].get("accuracy", 0) if self.best_model else 0

        summary = {
            "task": task,
            "dataset": dataset,
            "novel_model_metrics": self.best_model["metrics"] if self.best_model else {},
            "best_baseline_accuracy": self.best_baseline_acc,
            "novel_beats_baseline": best_acc >= self.best_baseline_acc,
            "iterations_used": len(self.all_models),
            "all_models": [
                {k: v for k, v in m.items()
                 if k not in ("train_losses", "val_losses")}
                for m in self.all_models
            ],
            "baselines_trained": len(self.baselines),
            "ablation_runs": len(ablations),
            "research_log": self.research_log,
            "report_path": str(self.output_dir / "research_paper.md"),
            "elapsed_s": round(elapsed),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        (self.output_dir / "pipeline_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )

        print(f"\n{'='*70}")
        print(f"  Research complete in {elapsed:.0f}s")
        print(f"  Novel model: {best_acc:.4f} accuracy "
              f"({'BEATS' if best_acc >= self.best_baseline_acc else 'behind'} "
              f"baselines at {self.best_baseline_acc:.4f})")
        print(f"  Iterations: {len(self.all_models)} models trained")
        print(f"  Report: {self.output_dir / 'research_paper.md'}")
        print(f"{'='*70}")

        return summary

    # ── LLM-driven decision making ──────────────────────────────────

    def _think(
        self,
        task: str,
        base_architecture: str,
        n_top_genes: int,
        n_classes: int,
        n_samples: int,
        sizing: dict,
        iteration: int,
    ) -> dict[str, Any]:
        """The agent's brain: analyze history, decide next action."""

        history = self._format_history()

        prompt = f"""You are an autonomous ML research agent. Your goal is to build a model
that BEATS the best baseline accuracy of {self.best_baseline_acc:.4f}.

TASK: {task}
DATASET: {n_samples} samples, {n_top_genes} features, {n_classes} classes
ITERATION: {iteration}/{self.max_iterations}
BEST MODEL SO FAR: {json.dumps(self.best_model.get('metrics', {}) if self.best_model else {}, default=str)}

AVAILABLE ARCHITECTURES: attention_gene_network, residual_mlp, gene_transformer,
  expression_vae, contrastive_encoder, multi_modal_encoder
AUTO-SIZED DEFAULTS: hidden_dim={sizing['hidden_dim']}, n_layers={sizing['n_layers']},
  dropout={sizing['dropout']}, batch_size={sizing['batch_size']}

PREVIOUS ATTEMPTS:
{history}

BASELINES:
{self._format_baselines()}

Analyze the training history carefully. Look at:
1. Training curves: is the model underfitting or overfitting?
2. Gap to baseline: how far are we?
3. What strategies have/haven't been tried?
4. Whether we need more regularization, different architecture, more features, etc.

Return JSON with your decision:
{{
  "action": "train_model" | "hyperparam_search" | "design_from_paper" | "stop",
  "reasoning": "2-3 sentences explaining your analysis and decision",
  "architecture": "<architecture name>",
  "config": {{
    "hidden_dim": <int>,
    "n_layers": <int>,
    "n_heads": <int, must divide hidden_dim>,
    "dropout": <float 0.0-0.5>,
    "input_dim": {n_top_genes}
  }},
  "train_params": {{
    "learning_rate": <float>,
    "weight_decay": <float>,
    "batch_size": <int>,
    "epochs": <int>,
    "patience": <int>,
    "scheduler": "cosine" | "plateau"
  }},
  "n_top_genes": <int, how many genes to use as features>
}}

If action is "hyperparam_search", also include:
  "search_space": {{
    "learning_rate": [<min>, <max>],
    "weight_decay": [<min>, <max>],
    "hidden_dim": [<option1>, <option2>, ...],
    "dropout": [<min>, <max>]
  }},
  "n_trials": <int, 3-8>

If action is "stop", explain why further iteration won't help.

IMPORTANT: n_heads MUST evenly divide hidden_dim. Valid: 32/4=8, 48/4=12, 64/8=8.
Return ONLY valid JSON."""

        try:
            text = self.llm.generate(
                prompt,
                system="You are an expert ML researcher. Analyze results carefully and make data-driven decisions.",
                temperature=0.3,
            )
            action = extract_json_object(text)
        except Exception as e:
            action = None

        if not action or "action" not in action:
            return self._fallback_action(
                base_architecture, n_top_genes, n_classes, sizing, iteration,
            )

        config = action.get("config", {})
        hd = config.get("hidden_dim", sizing["hidden_dim"])
        nh = config.get("n_heads", _valid_heads(hd))
        if hd % nh != 0:
            nh = _valid_heads(hd)
        config["n_heads"] = nh
        config.setdefault("input_dim", n_top_genes)
        config.setdefault("n_classes", n_classes)
        action["config"] = config

        return action

    def _fallback_action(
        self, arch: str, n_genes: int, n_classes: int,
        sizing: dict, iteration: int,
    ) -> dict[str, Any]:
        """Fallback when LLM response can't be parsed."""
        hd = sizing["hidden_dim"]
        strategies = [
            {"architecture": arch, "hidden_dim": hd, "dropout": sizing["dropout"],
             "lr": 1e-3, "desc": "Auto-sized baseline"},
            {"architecture": arch, "hidden_dim": hd, "dropout": min(sizing["dropout"] + 0.15, 0.5),
             "lr": 5e-4, "desc": "Higher regularization"},
            {"architecture": "residual_mlp", "hidden_dim": max(hd, 64), "dropout": 0.4,
             "lr": 1e-3, "desc": "Residual MLP fallback"},
            {"architecture": arch, "hidden_dim": max(hd // 2, 16), "dropout": 0.3,
             "lr": 5e-4, "desc": "Smaller model"},
        ]
        idx = min(iteration - 1, len(strategies) - 1)
        s = strategies[idx]
        return {
            "action": "train_model",
            "reasoning": f"Fallback strategy: {s['desc']}",
            "architecture": s["architecture"],
            "config": {
                "input_dim": n_genes, "n_classes": n_classes,
                "hidden_dim": s["hidden_dim"], "n_layers": sizing["n_layers"],
                "n_heads": _valid_heads(s["hidden_dim"]),
                "dropout": s["dropout"],
            },
            "train_params": {
                "learning_rate": s["lr"],
                "weight_decay": sizing["weight_decay"],
                "batch_size": sizing["batch_size"],
                "epochs": sizing["epochs"],
                "patience": sizing["patience"],
                "scheduler": "cosine",
            },
            "n_top_genes": n_genes,
        }

    def _format_baselines(self) -> str:
        bl = [{"name": b.name,
               "accuracy": b.metrics.get("accuracy", 0),
               "cv": b.metrics.get("cross_val_mean", 0)}
              for b in self.baselines]
        return json.dumps(bl, default=str)

    def _format_history(self) -> str:
        if not self.all_models:
            return "No previous attempts."
        lines = []
        for m in self.all_models:
            status = "✓" if m.get("accuracy", 0) >= self.best_baseline_acc else "✗"
            curves = ""
            tl = m.get("train_losses", [])
            vl = m.get("val_losses", [])
            if tl:
                curves = (f" train_loss: {tl[0]:.4f}->{tl[-1]:.4f},"
                          f" val_loss: {vl[0]:.4f}->{vl[-1]:.4f}" if vl else "")
            lines.append(
                f"  [{status}] {m.get('name', '?')}: "
                f"acc={m.get('accuracy', '?')}, f1={m.get('f1', '?')}, "
                f"epochs={m.get('epochs_trained', '?')}, "
                f"params={m.get('n_params', '?')}{curves}"
                f"\n      Config: hd={m.get('config', {}).get('hidden_dim', '?')}, "
                f"do={m.get('config', {}).get('dropout', '?')}, "
                f"lr={m.get('learning_rate', '?')}, wd={m.get('weight_decay', '?')}"
            )
            if m.get("error"):
                lines[-1] += f"\n      ERROR: {m['error'][:100]}"
        return "\n".join(lines)

    # ── Action execution ────────────────────────────────────────────

    def _execute_action(
        self,
        action: dict[str, Any],
        ctx: ToolContext,
        task: str,
        n_top_genes: int,
        n_classes: int,
        sizing: dict,
    ) -> dict[str, Any] | None:
        action_type = action.get("action", "train_model")

        if action_type == "train_model":
            return self._train_model(action, ctx, n_top_genes)
        elif action_type == "hyperparam_search":
            return self._hyperparam_search(action, ctx, n_top_genes, n_classes, sizing)
        elif action_type == "design_from_paper":
            return self._design_and_train_from_paper(action, ctx, task, n_top_genes)
        return None

    def _train_model(
        self, action: dict, ctx: ToolContext, n_top_genes: int,
    ) -> dict[str, Any]:
        arch = action.get("architecture", "attention_gene_network")
        config = action.get("config", {})
        tp = action.get("train_params", {})
        feat_genes = action.get("n_top_genes", n_top_genes)

        try:
            build = run_tool("build_architecture", ctx, {
                "architecture_type": arch,
                "config": config,
                "description": action.get("reasoning", ""),
            })
            if "error" in build:
                raise RuntimeError(build["error"])

            n_params = build.get("n_params", 0)
            print(f"  Built {arch}: {n_params:,} params")

            train = run_tool("train_model_pipeline", ctx, {
                "model_path": build["model_path"],
                "task": "classification",
                "n_top_genes": feat_genes,
                **tp,
            })
            if "error" in train:
                raise RuntimeError(train["error"])

            result = {
                "name": f"{arch} ({action.get('reasoning', '')[:50]})",
                "architecture": arch,
                "config": config,
                "accuracy": train.get("accuracy", 0),
                "f1": train.get("f1", 0),
                "best_val_loss": train.get("best_val_loss", float("inf")),
                "epochs_trained": train.get("epochs_trained", 0),
                "n_params": n_params,
                "model_path": train.get("model_path", ""),
                "train_losses": train.get("train_losses", []),
                "val_losses": train.get("val_losses", []),
                "learning_rate": tp.get("learning_rate"),
                "weight_decay": tp.get("weight_decay"),
            }
            self.all_models.append(result)
            self._log("train", f"{arch}: acc={result['accuracy']:.4f}", result)
            return result

        except Exception as e:
            err_result = {
                "name": arch, "error": str(e), "config": config,
                "accuracy": 0, "f1": 0,
            }
            self.all_models.append(err_result)
            self._log("train_error", str(e), err_result)
            return err_result

    def _hyperparam_search(
        self, action: dict, ctx: ToolContext,
        n_top_genes: int, n_classes: int, sizing: dict,
    ) -> dict[str, Any]:
        """Random hyperparameter search with LLM-defined search space."""
        search_space = action.get("search_space", {})
        n_trials = min(int(action.get("n_trials", 5)), 8)
        arch = action.get("architecture", "attention_gene_network")
        base_config = action.get("config", {})

        print(f"  Hyperparameter search: {n_trials} trials")

        best_trial: dict[str, Any] | None = None
        best_acc = 0.0

        for trial in range(n_trials):
            lr_range = search_space.get("learning_rate", [1e-4, 5e-3])
            wd_range = search_space.get("weight_decay", [1e-5, 1e-2])
            do_range = search_space.get("dropout", [0.1, 0.5])
            hd_options = search_space.get("hidden_dim", [sizing["hidden_dim"]])

            lr = 10 ** random.uniform(
                np.log10(lr_range[0]), np.log10(lr_range[1])
            )
            wd = 10 ** random.uniform(
                np.log10(max(wd_range[0], 1e-6)),
                np.log10(wd_range[1]),
            )
            do = random.uniform(do_range[0], do_range[1])
            hd = random.choice(hd_options) if isinstance(hd_options, list) else hd_options

            config = {
                **base_config,
                "hidden_dim": hd,
                "n_heads": _valid_heads(hd),
                "dropout": round(do, 3),
                "input_dim": n_top_genes,
                "n_classes": n_classes,
            }
            tp = {
                "learning_rate": round(lr, 6),
                "weight_decay": round(wd, 6),
                "batch_size": sizing["batch_size"],
                "epochs": min(sizing["epochs"], 200),
                "patience": sizing["patience"],
                "scheduler": random.choice(["cosine", "plateau"]),
            }

            trial_action = {
                "action": "train_model",
                "reasoning": f"HP search trial {trial+1}/{n_trials}: "
                             f"lr={lr:.2e}, wd={wd:.2e}, do={do:.2f}, hd={hd}",
                "architecture": arch,
                "config": config,
                "train_params": tp,
                "n_top_genes": n_top_genes,
            }

            result = self._train_model(trial_action, ctx, n_top_genes)
            acc = result.get("accuracy", 0)
            print(f"    Trial {trial+1}: lr={lr:.2e}, wd={wd:.2e}, "
                  f"do={do:.2f}, hd={hd} -> acc={acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_trial = result

            if acc >= self.best_baseline_acc:
                print(f"    HP search found winning config!")
                break

        return best_trial or {"accuracy": 0, "error": "All HP trials failed"}

    def _design_and_train_from_paper(
        self, action: dict, ctx: ToolContext,
        task: str, n_top_genes: int,
    ) -> dict[str, Any]:
        """Use design_from_paper tool to create architecture from literature."""
        paper_text = action.get("paper_text", "")
        if not paper_text:
            paper_text = (
                f"Design an architecture for {task} classification on "
                f"gene expression data with {n_top_genes} features."
            )

        try:
            result = run_tool("design_from_paper", ctx, {
                "paper_text": paper_text,
                "task": "classification",
                "input_dim": n_top_genes,
                "auto_train": True,
            })
            if "error" in result:
                raise RuntimeError(result["error"])

            tr = result.get("training_result", {})
            entry = {
                "name": f"Paper-inspired ({result.get('paper_architecture_type', '?')})",
                "architecture": result.get("paper_architecture_type", "unknown"),
                "config": result.get("paper_config", {}),
                "accuracy": tr.get("accuracy", 0),
                "f1": tr.get("f1", 0),
                "best_val_loss": tr.get("best_val_loss", float("inf")),
                "epochs_trained": tr.get("epochs_trained", 0),
                "n_params": result.get("n_params", 0),
                "model_path": tr.get("model_path", result.get("model_path", "")),
                "design_rationale": result.get("design_rationale", ""),
            }
            self.all_models.append(entry)
            self._log("design_from_paper", entry.get("design_rationale", ""), entry)
            return entry
        except Exception as e:
            err = {"error": str(e), "accuracy": 0}
            self._log("design_error", str(e), err)
            return err

    # ── K-fold cross-validation ─────────────────────────────────────

    def _kfold_cv(
        self,
        ctx: ToolContext,
        model_entry: dict[str, Any],
        n_top_genes: int,
        n_classes: int,
        k: int = 5,
    ) -> dict[str, Any] | None:
        """Run k-fold CV on the best model architecture."""
        import torch
        import torch.nn as nn
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        try:
            from src.tools.model_builder import _extract_features
            from src.tools.architecture_catalog import build_from_catalog

            X, y, gene_names = _extract_features(ctx, None, n_top_genes)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            config = model_entry.get("architecture_config", {})
            config["input_dim"] = X.shape[1]
            config["n_classes"] = n_classes

            arch_name = config.pop("architecture_type", None)
            if not arch_name:
                name = model_entry.get("name", "")
                for candidate in ["attention_gene_network", "residual_mlp",
                                   "gene_transformer", "expression_vae"]:
                    if candidate in name.lower():
                        arch_name = candidate
                        break
                if not arch_name:
                    mp = model_entry.get("model_path", "")
                    if mp and Path(mp).exists():
                        ckpt = torch.load(mp, map_location="cpu", weights_only=False)
                        arch_name = ckpt.get("architecture_type", "attention_gene_network")
                    else:
                        arch_name = "attention_gene_network"

            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            fold_accs: list[float] = []

            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])
                y_train, y_test = y[train_idx], y[test_idx]

                model = build_from_catalog(arch_name, config).to(device)
                X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
                y_tr = torch.tensor(y_train, dtype=torch.long, device=device)
                X_te = torch.tensor(X_test, dtype=torch.float32, device=device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
                criterion = nn.CrossEntropyLoss()
                bs = 16

                best_state = None
                best_loss = float("inf")
                patience_counter = 0

                for epoch in range(150):
                    model.train()
                    idx = torch.randperm(len(X_tr), device=device)
                    for start in range(0, len(X_tr), bs):
                        batch_idx = idx[start:start + bs]
                        optimizer.zero_grad()
                        loss = criterion(model(X_tr[batch_idx]), y_tr[batch_idx])
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        val_loss = criterion(model(X_te), torch.tensor(y_test, dtype=torch.long, device=device)).item()
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter >= 20:
                        break

                if best_state:
                    model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    preds = model(X_te).argmax(dim=1).cpu().numpy()

                from sklearn.metrics import accuracy_score
                fold_acc = float(accuracy_score(y_test, preds))
                fold_accs.append(fold_acc)
                print(f"    Fold {fold+1}/{k}: accuracy={fold_acc:.4f}")

            return {
                "cv_scores": fold_accs,
                "cv_mean": float(np.mean(fold_accs)),
                "cv_std": float(np.std(fold_accs)),
            }
        except Exception as e:
            print(f"  K-fold CV failed: {e}")
            return None

    # ── Ablation studies ────────────────────────────────────────────

    def _run_ablation_studies(
        self, ctx: ToolContext, n_classes: int, n_top_genes: int,
        sizing: dict, n_genes_total: int, n_samples: int,
    ) -> list[dict[str, Any]]:
        if not self.best_model:
            return []

        config = self.best_model.get("architecture_config", {})
        hd = config.get("hidden_dim", sizing["hidden_dim"])

        mp = self.best_model.get("model_path", "")
        if mp and Path(mp).exists():
            import torch
            ckpt = torch.load(mp, map_location="cpu", weights_only=False)
            arch_name = ckpt.get("architecture_type", "attention_gene_network")
        else:
            arch_name = "attention_gene_network"

        ablation_specs = [
            {"label": f"Half hidden ({hd//2})", "override": {"hidden_dim": max(hd // 2, 16)}},
            {"label": f"Double hidden ({hd*2})", "override": {"hidden_dim": hd * 2}},
            {"label": "Fewer genes (100)", "override": {"input_dim": min(100, n_genes_total)}, "n_genes": 100},
            {"label": "More genes (500)", "override": {"input_dim": min(500, n_genes_total)}, "n_genes": min(500, n_genes_total)},
            {"label": "No dropout", "override": {"dropout": 0.0}},
        ]

        ablations = []
        for spec in ablation_specs:
            abl_config = {
                **config,
                "n_classes": n_classes,
                "n_heads": _valid_heads(hd),
                **spec["override"],
            }
            abl_config["n_heads"] = _valid_heads(abl_config.get("hidden_dim", hd))
            abl_config.setdefault("input_dim", spec.get("n_genes", n_top_genes))
            abl_n_genes = spec.get("n_genes", n_top_genes)

            try:
                build = run_tool("build_architecture", ctx, {
                    "architecture_type": arch_name,
                    "config": abl_config,
                    "description": f"Ablation: {spec['label']}",
                })
                if "error" in build:
                    raise RuntimeError(build["error"])

                train = run_tool("train_model_pipeline", ctx, {
                    "model_path": build["model_path"],
                    "task": "classification",
                    "epochs": min(sizing["epochs"], 150),
                    "batch_size": sizing["batch_size"],
                    "learning_rate": 5e-4,
                    "patience": sizing["patience"],
                    "weight_decay": sizing["weight_decay"],
                    "n_top_genes": abl_n_genes,
                })
                metrics = {k: float(train[k]) for k in ("accuracy", "f1", "best_val_loss")
                           if k in train and isinstance(train[k], (int, float))}
                ablations.append({
                    "varied_params": spec["override"],
                    "metrics": metrics,
                    "label": spec["label"],
                })
                print(f"    {spec['label']}: acc={metrics.get('accuracy', '?'):.4f}")
            except Exception as e:
                print(f"    {spec['label']} failed: {e}")

        return ablations

    # ── Logging ─────────────────────────────────────────────────────

    def _log(self, event: str, message: str, data: dict[str, Any]) -> None:
        self.research_log.append({
            "event": event,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {k: v for k, v in data.items()
                     if k not in ("train_losses", "val_losses")},
        })


# ─── Backward-compatible alias ──────────────────────────────────────

ModelResearchPipeline = AutonomousResearchAgent
