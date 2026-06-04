"""End-to-end test of agentic-scientific-discovery model building tools.

Downloads a small GEO dataset, builds novel architectures, trains them,
and benchmarks the results — all using the pipeline's own tool functions.
"""

import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.tools.data_analysis import ToolContext
from src.tools.model_builder import build_architecture, train_model_pipeline, benchmark_model
from src.tools.architecture_catalog import ARCHITECTURE_CATALOG, CATALOG_BY_NAME

# ---------------------------------------------------------------------------
# Step 1: Download a small GEO dataset (GSE5281 — Alzheimer's brain regions)
# ---------------------------------------------------------------------------

def download_geo_data() -> tuple[pd.DataFrame, pd.Series]:
    """Download GSE5281 or generate realistic synthetic data as fallback."""
    print("=" * 70)
    print("STEP 1: Acquiring gene expression data")
    print("=" * 70)

    try:
        import GEOparse
        print("Downloading GSE5281 (Alzheimer's brain transcriptomics)...")
        gse = GEOparse.get_GEO("GSE5281", destdir="/tmp/geo_cache", silent=True)

        samples = {}
        groups = {}
        for gsm_name, gsm in gse.gsms.items():
            title = gsm.metadata.get("title", [""])[0].lower()
            if "control" in title or "normal" in title:
                group = "control"
            elif "alzheimer" in title or "ad" in title or "affected" in title:
                group = "alzheimer"
            else:
                continue
            expr = gsm.table
            if expr is not None and len(expr) > 0:
                expr = expr.set_index("ID_REF")["VALUE"]
                samples[gsm_name] = expr
                groups[gsm_name] = group

        if len(samples) >= 20:
            expression = pd.DataFrame(samples).dropna()
            group_series = pd.Series(groups)
            common = expression.columns.intersection(group_series.index)
            expression = expression[common]
            group_series = group_series[common]
            print(f"  Downloaded: {expression.shape[0]} probes x {expression.shape[1]} samples")
            print(f"  Groups: {group_series.value_counts().to_dict()}")
            return expression, group_series
    except Exception as e:
        print(f"  GEO download failed ({type(e).__name__}: {e}), using synthetic data")

    print("Generating realistic synthetic Alzheimer's gene expression data...")
    np.random.seed(42)
    n_genes = 2000
    n_controls = 40
    n_alzheimer = 40
    n_samples = n_controls + n_alzheimer

    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    real_gene_names = [
        "APP", "PSEN1", "PSEN2", "APOE", "MAPT", "BACE1", "CLU", "TREM2",
        "BIN1", "ABCA7", "CD33", "SORL1", "ADAM10", "PICALM", "CR1",
        "GSK3B", "CDK5", "NCSTN", "APH1A", "PSENEN", "IDE", "NEP",
        "BDNF", "NGF", "CREB1", "SYP", "DLG4", "SNAP25", "SYN1", "VAMP2",
    ]
    gene_names[:len(real_gene_names)] = real_gene_names

    base_expr = np.random.lognormal(mean=6, sigma=1.5, size=(n_genes, 1))
    noise = np.random.normal(0, 0.3, size=(n_genes, n_samples))
    expression_matrix = base_expr + noise

    n_de_genes = 200
    de_effect = np.random.choice([-1, 1], size=n_de_genes) * np.random.uniform(0.5, 2.0, size=n_de_genes)
    expression_matrix[:n_de_genes, n_controls:] += de_effect.reshape(-1, 1)

    sample_names = [f"CTL_{i+1}" for i in range(n_controls)] + \
                   [f"AD_{i+1}" for i in range(n_alzheimer)]
    group_labels = ["control"] * n_controls + ["alzheimer"] * n_alzheimer

    expression = pd.DataFrame(
        expression_matrix,
        index=gene_names,
        columns=sample_names,
    )
    group_series = pd.Series(group_labels, index=sample_names)

    print(f"  Generated: {expression.shape[0]} genes x {expression.shape[1]} samples")
    print(f"  Groups: {group_series.value_counts().to_dict()}")
    print(f"  Differentially expressed genes: {n_de_genes}")
    return expression, group_series


# ---------------------------------------------------------------------------
# Step 2: Create ToolContext
# ---------------------------------------------------------------------------

def create_context(expression: pd.DataFrame, groups: pd.Series) -> ToolContext:
    print("\n" + "=" * 70)
    print("STEP 2: Creating ToolContext")
    print("=" * 70)

    output_dir = Path("./outputs/test_model_building")
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = ToolContext(
        expression=expression,
        groups=groups,
        output_dir=output_dir,
    )

    var_genes = expression.var(axis=1).sort_values(ascending=False).head(200)
    print(f"  Top 200 genes by variance selected (range: {var_genes.iloc[-1]:.2f} - {var_genes.iloc[0]:.2f})")
    print(f"  Output directory: {output_dir}")
    return ctx


# ---------------------------------------------------------------------------
# Step 3: Test build_architecture — attention_gene_network
# ---------------------------------------------------------------------------

def test_attention_network(ctx: ToolContext) -> dict:
    print("\n" + "=" * 70)
    print("STEP 3: Build Architecture — attention_gene_network")
    print("=" * 70)

    n_features = 200
    n_classes = len(ctx.groups.unique())

    result = build_architecture(ctx, {
        "architecture_type": "attention_gene_network",
        "config": {
            "input_dim": n_features,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 3,
            "n_classes": n_classes,
            "dropout": 0.1,
            "max_len": 500,
        },
        "description": "Self-attention network for Alzheimer's gene expression classification",
    })

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return result

    print(f"  Model ID: {result['model_id']}")
    print(f"  Architecture: {result['architecture']}")
    print(f"  Parameters: {result['n_params']:,} ({result['n_trainable_params']:,} trainable)")
    print(f"  Architecture hash: {result['arch_hash']}")
    print(f"  Device: {result['device']}")
    print(f"  Saved to: {result['model_path']}")

    checkpoint = torch.load(result['model_path'], map_location='cpu', weights_only=False)
    from src.tools.architecture_catalog import build_from_catalog
    model = build_from_catalog("attention_gene_network", checkpoint['config'])
    print(f"\n  Architecture structure:")
    print(f"  {model}")
    return result


# ---------------------------------------------------------------------------
# Step 4: Test build_architecture — graph_neural_network (GCN)
# ---------------------------------------------------------------------------

def test_gnn_architecture(ctx: ToolContext) -> dict:
    print("\n" + "=" * 70)
    print("STEP 4: Build Architecture — graph_neural_network (GCN)")
    print("=" * 70)

    n_features = 200
    n_classes = len(ctx.groups.unique())

    result = build_architecture(ctx, {
        "architecture_type": "gcn_message_passing",
        "config": {
            "input_dim": n_features,
            "hidden_dim": 64,
            "n_layers": 3,
            "n_classes": n_classes,
            "dropout": 0.1,
            "task": "graph_classification",
        },
        "description": "Graph Convolutional Network for gene co-expression network analysis",
    })

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return result

    print(f"  Model ID: {result['model_id']}")
    print(f"  Architecture: {result['architecture']}")
    print(f"  Parameters: {result['n_params']:,} ({result['n_trainable_params']:,} trainable)")
    print(f"  Architecture hash: {result['arch_hash']}")
    print(f"  Device: {result['device']}")
    print(f"  Saved to: {result['model_path']}")

    checkpoint = torch.load(result['model_path'], map_location='cpu', weights_only=False)
    from src.tools.architecture_catalog import build_from_catalog
    model = build_from_catalog("gcn_message_passing", checkpoint['config'])
    print(f"\n  Architecture structure:")
    print(f"  {model}")
    return result


# ---------------------------------------------------------------------------
# Step 5: Train the attention model
# ---------------------------------------------------------------------------

def test_training(ctx: ToolContext, attention_result: dict) -> dict:
    print("\n" + "=" * 70)
    print("STEP 5: Train attention_gene_network on expression data")
    print("=" * 70)

    if "error" in attention_result:
        print("  SKIPPED: architecture build failed")
        return {"error": "skipped due to build failure"}

    result = train_model_pipeline(ctx, {
        "model_path": attention_result["model_path"],
        "epochs": 10,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "patience": 8,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "scheduler": "cosine",
        "n_top_genes": 200,
        "test_fraction": 0.2,
        "task": "classification",
    })

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return result

    print(f"\n  Training Results:")
    print(f"  Model ID: {result['model_id']}")
    print(f"  Epochs trained: {result['epochs_trained']}")
    print(f"  Best validation loss: {result['best_val_loss']:.4f}")
    print(f"  Final accuracy: {result.get('accuracy', 'N/A')}")
    print(f"  Final F1 score: {result.get('f1', 'N/A')}")
    print(f"  Training time: {result['training_time_s']:.1f}s")
    print(f"  Device: {result['device']} (AMP: {result['mixed_precision']})")
    print(f"  Samples: {result['n_samples']} | Features: {result['n_features']}")
    print(f"  Saved to: {result['model_path']}")

    print(f"\n  Loss trajectory (train):")
    for i, l in enumerate(result['train_losses']):
        bar = "#" * int(l * 20)
        print(f"    Epoch {i+1:2d}: {l:.4f} {bar}")

    print(f"\n  Loss trajectory (val):")
    for i, l in enumerate(result['val_losses']):
        bar = "#" * int(l * 20)
        print(f"    Epoch {i+1:2d}: {l:.4f} {bar}")

    return result


# ---------------------------------------------------------------------------
# Step 6: Benchmark the trained model
# ---------------------------------------------------------------------------

def test_benchmark(ctx: ToolContext, train_result: dict) -> dict:
    print("\n" + "=" * 70)
    print("STEP 6: Benchmark trained model")
    print("=" * 70)

    if "error" in train_result:
        print("  SKIPPED: training failed")
        return {"error": "skipped due to training failure"}

    result = benchmark_model(ctx, {
        "model_path": train_result["model_path"],
        "metrics": ["accuracy", "f1", "auc_roc", "precision", "recall"],
        "n_top_genes": 200,
        "cross_validate": True,
        "cv_folds": 5,
    })

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return result

    primary = result.get("primary_evaluation", {})
    print(f"\n  Benchmark Results:")
    print(f"  Architecture: {primary.get('architecture', 'N/A')}")
    print(f"  Accuracy: {primary.get('accuracy', 'N/A')}")
    print(f"  F1 Score: {primary.get('f1', 'N/A')}")
    print(f"  AUC-ROC: {primary.get('auc_roc', 'N/A')}")
    print(f"  Precision: {primary.get('precision', 'N/A')}")
    print(f"  Recall: {primary.get('recall', 'N/A')}")

    if "cv_scores" in primary:
        print(f"\n  Cross-validation (5-fold):")
        for i, score in enumerate(primary['cv_scores']):
            print(f"    Fold {i+1}: {score:.4f}")
        print(f"    Mean: {primary['cv_mean']:.4f} +/- {primary['cv_std']:.4f}")

    print(f"\n  Dataset: {primary.get('n_samples', '?')} samples x {primary.get('n_features', '?')} features")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "#" * 70)
    print("#  AGENTIC SCIENTIFIC DISCOVERY — MODEL BUILDING TEST")
    print("#  Testing novel architecture creation and training pipeline")
    print("#" * 70)
    print(f"\n  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\n  Architecture catalog: {len(ARCHITECTURE_CATALOG)} patterns")
    print(f"  Available: {list(CATALOG_BY_NAME.keys())}")

    t_total = time.time()

    expression, groups = download_geo_data()
    ctx = create_context(expression, groups)
    attention_result = test_attention_network(ctx)
    gnn_result = test_gnn_architecture(ctx)
    train_result = test_training(ctx, attention_result)
    benchmark_result = test_benchmark(ctx, train_result)

    elapsed = time.time() - t_total

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print(f"\n  Architectures built:")
    if "error" not in attention_result:
        print(f"    1. attention_gene_network — {attention_result['n_params']:,} params")
    if "error" not in gnn_result:
        print(f"    2. gcn_message_passing — {gnn_result['n_params']:,} params")

    if "error" not in train_result:
        print(f"\n  Training:")
        print(f"    Epochs: {train_result['epochs_trained']}")
        print(f"    Best val loss: {train_result['best_val_loss']:.4f}")
        print(f"    Accuracy: {train_result.get('accuracy', 'N/A')}")
        print(f"    F1: {train_result.get('f1', 'N/A')}")

    if "error" not in benchmark_result:
        primary = benchmark_result.get("primary_evaluation", {})
        print(f"\n  Benchmark:")
        print(f"    Accuracy: {primary.get('accuracy', 'N/A')}")
        print(f"    CV Mean: {primary.get('cv_mean', 'N/A')}")
        print(f"    CV Std: {primary.get('cv_std', 'N/A')}")

    all_passed = all("error" not in r for r in [attention_result, gnn_result, train_result, benchmark_result])
    print(f"\n  Pipeline status: {'ALL PASSED' if all_passed else 'SOME FAILURES'}")
    print("\n" + "#" * 70)
    print("#  TEST COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    main()
