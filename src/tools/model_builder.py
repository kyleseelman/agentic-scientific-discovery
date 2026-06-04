"""Novel model-building tools for the research agent.

Design, train, and benchmark custom PyTorch architectures informed by
AI+bio literature — from LoRA fine-tuning of protein/genomic LMs to
generating custom architectures from paper insights.

All tools follow the TOOL_REGISTRY signature:
``(ctx: ToolContext, params: dict) -> dict``.

Heavy imports (torch, transformers, peft) are deferred to call-time so
the module loads quickly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.tools.data_analysis import ToolContext

logger = logging.getLogger(__name__)

_MODEL_STORE = None


def _get_model_store():
    global _MODEL_STORE
    if _MODEL_STORE is None:
        from src.memory.model_store import ModelStore
        _MODEL_STORE = ModelStore(Path("./outputs/model_registry.json"))
    return _MODEL_STORE


def _track_model(
    result: dict[str, Any],
    task: str,
    hypothesis_id: str | None = None,
    parent_model: str | None = None,
    paper_inspiration: str | None = None,
) -> None:
    try:
        from src.memory.model_store import ModelRecord
        store = _get_model_store()
        notes_parts = [task]
        if parent_model:
            notes_parts.append(f"parent={parent_model}")
        if paper_inspiration:
            notes_parts.append(f"paper={paper_inspiration}")
        store.add(ModelRecord(
            id=result.get("model_id", _model_id()),
            model_type=result.get("model_type", result.get("architecture", "unknown")),
            task=task,
            hypothesis_id=hypothesis_id,
            experiment_id=None,
            metrics={k: v for k, v in result.items()
                     if k in ("accuracy", "f1", "auc_roc", "loss", "val_loss",
                              "best_val_loss", "final_loss")
                     and isinstance(v, (int, float))},
            hyperparameters={k: v for k, v in result.items()
                            if k in ("n_features", "n_samples", "n_params",
                                     "architecture", "learning_rate", "epochs")},
            feature_genes=result.get("gene_set", []),
            model_path=result.get("model_path", ""),
            training_time_s=result.get("training_time_s", 0.0),
            notes="; ".join(notes_parts),
        ))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_id() -> str:
    return "mdl_" + hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]


def _ensure_output(ctx: ToolContext) -> Path:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    return ctx.output_dir


def _get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_python(obj: Any) -> Any:
    """Recursively convert numpy/torch types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
    except ImportError:
        pass
    return obj


def _extract_features(
    ctx: ToolContext,
    gene_subset: list[str] | None,
    n_top_genes: int = 500,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract (X, y, gene_names) from ToolContext.  X is samples-by-genes."""
    expr = ctx.expression
    if gene_subset:
        available = [g for g in gene_subset if g in expr.index]
        if not available:
            raise ValueError(f"None of the specified genes found. First 10: {list(expr.index[:10])}")
        expr = expr.loc[available]
    else:
        var = expr.var(axis=1).sort_values(ascending=False)
        expr = expr.loc[var.head(n_top_genes).index]

    X = expr.T.astype(float).values
    gene_names = list(expr.index.astype(str))

    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(ctx.groups.values)
    return X, y, gene_names


# ---------------------------------------------------------------------------
# Tool 1: build_architecture
# ---------------------------------------------------------------------------

def build_architecture(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Build and instantiate a PyTorch nn.Module from an architecture spec.

    params:
        architecture_type: str — one of: "gene_transformer",
            "protein_interaction_gat", "multi_modal_encoder", "residual_mlp",
            "contrastive_encoder", "expression_vae", "gcn_message_passing",
            "attention_gene_network", "custom"
        config: dict — architecture-specific configuration
            (input_dim, hidden_dim, n_heads, n_layers, n_classes, dropout, etc.)
        custom_code: str | None — PyTorch code string (only for type="custom")
        description: str — human-readable description of the architecture
    """
    import torch
    from src.tools.architecture_catalog import CATALOG_BY_NAME, build_from_catalog

    arch_type = str(params.get("architecture_type", "residual_mlp"))
    config = dict(params.get("config", {}))
    description = str(params.get("description", f"Built {arch_type} architecture"))
    custom_code = params.get("custom_code")

    device = _get_device()
    mid = _model_id()
    out_dir = _ensure_output(ctx)

    if arch_type == "custom" and custom_code:
        try:
            _ALLOWED_IMPORTS = {"torch", "torch.nn", "math", "numpy"}
            for line in custom_code.split("\n"):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    mod = stripped.split()[1].split(".")[0]
                    if mod not in _ALLOWED_IMPORTS and f"{mod}" not in {"np", "nn"}:
                        return {"error": f"Disallowed import: {mod}. Allowed: {_ALLOWED_IMPORTS}"}

            namespace: dict[str, Any] = {"torch": torch, "nn": torch.nn, "math": __import__("math")}
            exec(custom_code, namespace)

            model_cls = None
            for v in namespace.values():
                if isinstance(v, type) and issubclass(v, torch.nn.Module) and v is not torch.nn.Module:
                    model_cls = v
                    break
            if model_cls is None:
                return {"error": "Custom code must define an nn.Module subclass"}

            model = model_cls(**config).to(device)
        except Exception as e:
            return {"error": f"Custom architecture build failed: {type(e).__name__}: {e}"}
    elif arch_type in CATALOG_BY_NAME:
        try:
            model = build_from_catalog(arch_type, config).to(device)
        except Exception as e:
            return {"error": f"Failed to build {arch_type}: {type(e).__name__}: {e}"}
    else:
        available = list(CATALOG_BY_NAME.keys()) + ["custom"]
        return {"error": f"Unknown architecture_type: {arch_type}. Available: {available}"}

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    arch_hash = hashlib.sha256(str(model).encode()).hexdigest()[:12]

    model_path = out_dir / f"{mid}_{arch_type}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "architecture_type": arch_type,
        "config": config,
        "description": description,
        "arch_hash": arch_hash,
    }, model_path)

    print(f"Built {arch_type}: {n_params:,} params ({n_trainable:,} trainable) on {device}")

    result = _to_python({
        "model_id": mid,
        "architecture": arch_type,
        "model_type": arch_type,
        "description": description,
        "n_params": n_params,
        "n_trainable_params": n_trainable,
        "arch_hash": arch_hash,
        "config": config,
        "model_path": str(model_path),
        "device": str(device),
    })
    _track_model(result, "architecture_build")
    return result


# ---------------------------------------------------------------------------
# Tool 2: train_model_pipeline
# ---------------------------------------------------------------------------

def train_model_pipeline(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Full training pipeline with early stopping, LR scheduling, and AMP.

    Works with any nn.Module produced by build_architecture or other tools.

    params:
        model_path: str — path to .pt file from build_architecture
        epochs: int — max training epochs (default 50)
        learning_rate: float — initial LR (default 1e-3)
        batch_size: int — (default 32)
        patience: int — early stopping patience (default 10)
        weight_decay: float — AdamW weight decay (default 1e-4)
        grad_clip: float — gradient clipping max norm (default 1.0)
        scheduler: str — "cosine", "plateau", "step" (default "cosine")
        gene_subset: list[str] | None — genes to use as features
        n_top_genes: int — fallback top-N by variance (default 500)
        test_fraction: float — held-out test fraction (default 0.2)
        task: str — "classification" or "regression" (default "classification")
    """
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    model_path_str = str(params.get("model_path", ""))
    if not model_path_str or not Path(model_path_str).exists():
        return {"error": f"model_path required and must exist: {model_path_str}"}

    epochs = int(params.get("epochs", 50))
    lr = float(params.get("learning_rate", 1e-3))
    batch_size = int(params.get("batch_size", 32))
    patience = int(params.get("patience", 10))
    weight_decay = float(params.get("weight_decay", 1e-4))
    grad_clip = float(params.get("grad_clip", 1.0))
    sched_type = str(params.get("scheduler", "cosine"))
    gene_subset = params.get("gene_subset")
    n_top_genes = int(params.get("n_top_genes", 500))
    test_fraction = float(params.get("test_fraction", 0.2))
    task = str(params.get("task", "classification"))

    device = _get_device()
    checkpoint = torch.load(model_path_str, map_location=device, weights_only=False)
    arch_type = checkpoint.get("architecture_type", "unknown")
    config = checkpoint.get("config", {})

    from src.tools.architecture_catalog import build_from_catalog, CATALOG_BY_NAME
    if arch_type in CATALOG_BY_NAME:
        model = build_from_catalog(arch_type, config).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        return {"error": f"Cannot rebuild architecture {arch_type}. Use a catalog architecture."}

    X, y, gene_names = _extract_features(ctx, gene_subset, n_top_genes)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_fraction, random_state=42, stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_fraction, random_state=42,
        )

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.long if task == "classification" else torch.float32, device=device)
    X_te = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_te = torch.tensor(y_test, dtype=torch.long if task == "classification" else torch.float32, device=device)

    n_classes = len(np.unique(y))
    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience // 2)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1))

    use_amp = device.type == "cuda"
    scaler_amp = torch.amp.GradScaler("cuda") if use_amp else None

    print(f"Training {arch_type} | {X_train.shape[0]} train, {X_test.shape[0]} test | "
          f"device={device} | AMP={use_amp}")

    t0 = time.time()
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_tr), device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(X_tr), batch_size):
            idx = indices[start: start + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(xb)
                    loss = criterion(out, yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        model.eval()
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    val_out = model(X_te)
                    val_loss = criterion(val_out, y_te).item()
            else:
                val_out = model(X_te)
                val_loss = criterion(val_out, y_te).item()
        val_losses.append(val_loss)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}  train_loss={avg_train:.4f}  "
                  f"val_loss={val_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    training_time = time.time() - t0

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    metrics: dict[str, Any] = {"best_val_loss": best_val_loss}

    if task == "classification":
        from sklearn.metrics import accuracy_score, f1_score
        with torch.no_grad():
            logits = model(X_te)
            preds = logits.argmax(dim=1).cpu().numpy()
        metrics["accuracy"] = float(accuracy_score(y_test, preds))
        metrics["f1"] = float(f1_score(y_test, preds, average="weighted"))

    mid = _model_id()
    out_dir = _ensure_output(ctx)
    save_path = out_dir / f"{mid}_{arch_type}_trained.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "architecture_type": arch_type,
        "config": config,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "gene_names": gene_names,
        "n_classes": n_classes,
        "metrics": metrics,
    }, save_path)

    print(f"Training complete in {training_time:.1f}s — best_val_loss={best_val_loss:.4f}")

    result = _to_python({
        "model_id": mid,
        "architecture": arch_type,
        "model_type": arch_type,
        "model_path": str(save_path),
        "training_time_s": training_time,
        "epochs_trained": len(train_losses),
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "learning_rate": lr,
        "device": str(device),
        "mixed_precision": use_amp,
        **metrics,
    })
    _track_model(result, task, params.get("hypothesis_id"))
    return result


# ---------------------------------------------------------------------------
# Tool 3: finetune_protein_lm
# ---------------------------------------------------------------------------

def finetune_protein_lm(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Fine-tune ESM-2 for protein tasks using LoRA (parameter-efficient).

    params:
        base_model: str — ESM-2 variant (default "facebook/esm2_t6_8M_UR50D")
        sequences: list[str] — protein sequences
        labels: list[int|float] — labels for each sequence
        task: str — "classification", "regression", "ppi" (default "classification")
        lora_r: int — LoRA rank (default 8)
        lora_alpha: int — LoRA alpha (default 16)
        epochs: int — (default 5)
        batch_size: int — (default 8)
        learning_rate: float — (default 2e-4)
        max_length: int — max sequence length (default 512)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return {"error": "transformers is required. Run: pip install transformers"}

    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        return {"error": "peft is required for LoRA fine-tuning. Run: pip install peft"}

    base_model = str(params.get("base_model", "facebook/esm2_t6_8M_UR50D"))
    sequences = list(params.get("sequences", []))
    labels = list(params.get("labels", []))
    task = str(params.get("task", "classification"))
    lora_r = int(params.get("lora_r", 8))
    lora_alpha = int(params.get("lora_alpha", 16))
    epochs_count = int(params.get("epochs", 5))
    batch_size = int(params.get("batch_size", 8))
    lr = float(params.get("learning_rate", 2e-4))
    max_length = int(params.get("max_length", 512))

    if not sequences or not labels:
        return {"error": "sequences and labels are required (non-empty lists)"}
    if len(sequences) != len(labels):
        return {"error": f"sequences ({len(sequences)}) and labels ({len(labels)}) must match"}

    device = _get_device()
    n_classes = len(set(labels)) if task == "classification" else 1
    problem_type = "single_label_classification" if task == "classification" else "regression"

    print(f"Fine-tuning {base_model} with LoRA (r={lora_r}) for {task}")
    print(f"  {len(sequences)} sequences, device={device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=n_classes, problem_type=problem_type,
        )
    except Exception as e:
        return {"error": f"Failed to load {base_model}: {type(e).__name__}: {e}"}

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

    encoded = tokenizer(
        sequences, truncation=True, padding=True, max_length=max_length, return_tensors="pt",
    )
    label_dtype = torch.long if task == "classification" else torch.float32
    label_tensor = torch.tensor(labels, dtype=label_dtype)

    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"], label_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    t0 = time.time()
    loss_history: list[float] = []

    for epoch in range(epochs_count):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for input_ids, attn_mask, batch_labels in loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=batch_labels)
            outputs.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += outputs.loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch+1}/{epochs_count}  loss={avg_loss:.4f}")

    training_time = time.time() - t0

    mid = _model_id()
    out_dir = _ensure_output(ctx)
    model_dir = out_dir / f"{mid}_protein_lm"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    print(f"Protein LM fine-tuned in {training_time:.1f}s → {model_dir}")

    result = _to_python({
        "model_id": mid,
        "model_type": f"esm2_lora_{task}",
        "base_model": base_model,
        "task": task,
        "lora_rank": lora_r,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 1),
        "training_loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else None,
        "model_path": str(model_dir),
        "training_time_s": training_time,
        "n_sequences": len(sequences),
        "device": str(device),
    })
    _track_model(result, f"protein_lm_{task}", params.get("hypothesis_id"))
    return result


# ---------------------------------------------------------------------------
# Tool 4: finetune_genomic_lm
# ---------------------------------------------------------------------------

def finetune_genomic_lm(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Fine-tune any HuggingFace LM with LoRA/PEFT for genomic/biomedical tasks.

    params:
        base_model: str — HuggingFace model ID
        texts: list[str] — training texts
        labels: list[int|str] — labels
        task: str — "classification", "ner" (default "classification")
        lora_r: int — LoRA rank (default 8)
        lora_alpha: int — LoRA alpha (default 16)
        epochs: int — (default 3)
        batch_size: int — (default 16)
        learning_rate: float — (default 2e-5)
        max_length: int — (default 256)
        load_in_4bit: bool — quantized loading (default False)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return {"error": "transformers is required"}

    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        return {"error": "peft is required for LoRA. Run: pip install peft"}

    base_model = str(params.get("base_model", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"))
    texts = list(params.get("texts", []))
    labels_raw = list(params.get("labels", []))
    task = str(params.get("task", "classification"))
    lora_r = int(params.get("lora_r", 8))
    lora_alpha = int(params.get("lora_alpha", 16))
    epochs_count = int(params.get("epochs", 3))
    batch_size = int(params.get("batch_size", 16))
    lr = float(params.get("learning_rate", 2e-5))
    max_length = int(params.get("max_length", 256))
    load_in_4bit = bool(params.get("load_in_4bit", False))

    if not texts or not labels_raw:
        return {"error": "texts and labels are required"}
    if len(texts) != len(labels_raw):
        return {"error": f"texts ({len(texts)}) and labels ({len(labels_raw)}) must match"}

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)
    label_names = list(le.classes_)
    n_classes = len(label_names)

    device = _get_device()
    print(f"Fine-tuning {base_model} with LoRA for {task} ({n_classes} classes)")

    model_kwargs: dict[str, Any] = {}
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            print("  Loading in 4-bit quantization")
        except ImportError:
            print("  bitsandbytes not available, loading in full precision")

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=n_classes, **model_kwargs,
        )
    except Exception as e:
        return {"error": f"Failed to load {base_model}: {type(e).__name__}: {e}"}

    lora_target = ["query", "key", "value"]
    try:
        named_modules = [n for n, _ in model.named_modules()]
        if any("q_proj" in n for n in named_modules):
            lora_target = ["q_proj", "v_proj"]
    except Exception:
        pass

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=lora_target,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if not load_in_4bit:
        model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

    encoded = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    label_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"], label_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    t0 = time.time()
    loss_history: list[float] = []

    for epoch in range(epochs_count):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for input_ids, attn_mask, batch_labels in loader:
            actual_device = next(model.parameters()).device
            input_ids = input_ids.to(actual_device)
            attn_mask = attn_mask.to(actual_device)
            batch_labels = batch_labels.to(actual_device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=batch_labels)
            outputs.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += outputs.loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch+1}/{epochs_count}  loss={avg_loss:.4f}")

    training_time = time.time() - t0

    mid = _model_id()
    out_dir = _ensure_output(ctx)
    model_dir = out_dir / f"{mid}_genomic_lm"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    print(f"Genomic LM fine-tuned in {training_time:.1f}s → {model_dir}")

    result = _to_python({
        "model_id": mid,
        "model_type": f"lora_{base_model.split('/')[-1]}",
        "base_model": base_model,
        "task": task,
        "label_names": label_names,
        "n_classes": n_classes,
        "lora_rank": lora_r,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 1),
        "training_loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else None,
        "model_path": str(model_dir),
        "training_time_s": training_time,
        "n_texts": len(texts),
        "device": str(device),
    })
    _track_model(result, f"genomic_lm_{task}", params.get("hypothesis_id"))
    return result


# ---------------------------------------------------------------------------
# Tool 5: build_graph_model
# ---------------------------------------------------------------------------

def build_graph_model(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Build and optionally train a GNN for biological networks.

    Uses manual message-passing by default, with optional PyG backend.

    params:
        architecture: str — "gcn", "gat", "graphsage" (default "gcn")
        adjacency: list[list[float]] | str — adjacency matrix or path to CSV
        node_features: list[list[float]] | None — node feature matrix
            (default: uses expression data for nodes matching gene names)
        node_labels: list[int] | None — labels for node classification
        task: str — "node_classification", "link_prediction",
            "graph_classification" (default "node_classification")
        hidden_dim: int — (default 64)
        n_layers: int — (default 2)
        n_heads: int — GAT heads (default 4)
        epochs: int — training epochs (default 50)
        learning_rate: float — (default 1e-3)
        use_pyg: bool — attempt PyTorch Geometric (default False)
        gene_names: list[str] | None — gene/node identifiers
    """
    import torch
    import torch.nn as nn

    architecture = str(params.get("architecture", "gcn"))
    task = str(params.get("task", "node_classification"))
    hidden_dim = int(params.get("hidden_dim", 64))
    n_layers = int(params.get("n_layers", 2))
    n_heads = int(params.get("n_heads", 4))
    epochs_count = int(params.get("epochs", 50))
    lr = float(params.get("learning_rate", 1e-3))
    use_pyg = bool(params.get("use_pyg", False))
    gene_names_param = params.get("gene_names")

    device = _get_device()

    adj_param = params.get("adjacency")
    if adj_param is None:
        return {"error": "adjacency is required (matrix or path to CSV)"}

    if isinstance(adj_param, str) and Path(adj_param).exists():
        adj_df = pd.read_csv(adj_param, index_col=0)
        adj_np = adj_df.values.astype(float)
        if gene_names_param is None:
            gene_names_param = list(adj_df.index.astype(str))
    elif isinstance(adj_param, list):
        adj_np = np.array(adj_param, dtype=float)
    else:
        return {"error": "adjacency must be a matrix (list of lists) or a path to a CSV file"}

    n_nodes = adj_np.shape[0]

    node_features_param = params.get("node_features")
    if node_features_param is not None:
        X_nodes = np.array(node_features_param, dtype=float)
    elif gene_names_param:
        available = [g for g in gene_names_param if g in ctx.expression.index]
        if available:
            X_nodes = ctx.expression.loc[available].T.mean(axis=0).values.reshape(-1, 1)
            X_nodes = np.broadcast_to(X_nodes, (n_nodes, max(1, X_nodes.shape[-1])))
            X_nodes = np.ascontiguousarray(X_nodes)
        else:
            X_nodes = np.eye(n_nodes, dtype=float)
    else:
        X_nodes = np.eye(n_nodes, dtype=float)

    if X_nodes.shape[0] != n_nodes:
        X_nodes = np.eye(n_nodes, dtype=float)

    input_dim = X_nodes.shape[1]

    node_labels_param = params.get("node_labels")
    if node_labels_param is not None:
        node_labels = np.array(node_labels_param, dtype=int)
    else:
        node_labels = np.zeros(n_nodes, dtype=int)

    n_classes = max(len(set(node_labels)), 2)

    pyg_used = False
    if use_pyg:
        try:
            from torch_geometric.nn import GATConv, GCNConv, SAGEConv
            from torch_geometric.data import Data

            edge_index = torch.tensor(np.array(np.nonzero(adj_np)), dtype=torch.long)
            data = Data(
                x=torch.tensor(X_nodes, dtype=torch.float32),
                edge_index=edge_index,
                y=torch.tensor(node_labels, dtype=torch.long),
            ).to(device)

            ConvClass = {"gcn": GCNConv, "gat": GATConv, "graphsage": SAGEConv}
            conv_cls = ConvClass.get(architecture, GCNConv)

            class PyGModel(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.convs = nn.ModuleList()
                    dims = [input_dim] + [hidden_dim] * n_layers
                    for i in range(n_layers):
                        if architecture == "gat" and i == 0:
                            self.convs.append(conv_cls(dims[i], hidden_dim, heads=n_heads, concat=False))
                        else:
                            self.convs.append(conv_cls(dims[i], dims[i+1]))
                    self.head = nn.Linear(hidden_dim, n_classes)

                def forward(self, data: Any) -> torch.Tensor:
                    x, edge_index = data.x, data.edge_index
                    for conv in self.convs:
                        x = torch.relu(conv(x, edge_index))
                    return self.head(x)

            model = PyGModel().to(device)
            pyg_used = True
            print(f"Using PyTorch Geometric {architecture}")
        except ImportError:
            print("PyG not available, falling back to manual message-passing")

    if not pyg_used:
        from src.tools.architecture_catalog import build_from_catalog
        config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "n_classes": n_classes,
            "task": task,
            "dropout": 0.1,
        }
        if architecture == "gat":
            config["n_heads"] = n_heads
            model = build_from_catalog("protein_interaction_gat", config).to(device)
        else:
            model = build_from_catalog("gcn_message_passing", config).to(device)

    adj_t = torch.tensor(adj_np, dtype=torch.float32, device=device)
    X_t = torch.tensor(X_nodes, dtype=torch.float32, device=device)
    y_t = torch.tensor(node_labels, dtype=torch.long, device=device)

    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"Training {architecture} GNN: {n_nodes} nodes, {int(adj_np.sum())} edges, "
          f"{n_params:,} params")

    t0 = time.time()
    loss_history: list[float] = []

    for epoch in range(epochs_count):
        model.train()
        optimizer.zero_grad()

        if pyg_used:
            out = model(data)
        else:
            out = model(X_t, adj_t)

        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs_count}  loss={loss.item():.4f}")

    training_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        if pyg_used:
            logits = model(data)
        else:
            logits = model(X_t, adj_t)
        preds = logits.argmax(dim=1).cpu().numpy()

    if node_labels_param is not None:
        from sklearn.metrics import accuracy_score, f1_score
        acc = float(accuracy_score(node_labels, preds))
        f1 = float(f1_score(node_labels, preds, average="weighted", zero_division=0))
    else:
        acc = None
        f1 = None

    mid = _model_id()
    out_dir = _ensure_output(ctx)
    model_path = out_dir / f"{mid}_{architecture}_gnn.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "architecture": architecture,
        "config": {"input_dim": input_dim, "hidden_dim": hidden_dim,
                   "n_layers": n_layers, "n_classes": n_classes, "n_heads": n_heads},
        "gene_names": gene_names_param,
        "pyg_used": pyg_used,
    }, model_path)

    print(f"GNN trained in {training_time:.1f}s | acc={acc}")

    result = _to_python({
        "model_id": mid,
        "model_type": f"{architecture}_gnn",
        "architecture": architecture,
        "task": task,
        "n_nodes": n_nodes,
        "n_edges": int(adj_np.sum()),
        "n_params": n_params,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "pyg_used": pyg_used,
        "accuracy": acc,
        "f1": f1,
        "training_loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else None,
        "model_path": str(model_path),
        "training_time_s": training_time,
        "device": str(device),
    })
    _track_model(result, f"gnn_{task}", params.get("hypothesis_id"))
    return result


# ---------------------------------------------------------------------------
# Tool 6: design_from_paper
# ---------------------------------------------------------------------------

def design_from_paper(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Design and build an architecture from a paper description using LLM.

    Given a paper abstract or method description, uses the LLM to extract the
    architecture design, then calls build_architecture to instantiate it.

    params:
        paper_text: str — abstract or method section text
        task: str — target task (default "classification")
        input_dim: int — input feature dimension (default: auto from data)
        n_classes: int — number of output classes (default: auto from data)
        auto_train: bool — automatically train after building (default False)
    """
    paper_text = str(params.get("paper_text", ""))
    if not paper_text:
        return {"error": "paper_text is required (abstract or method description)"}

    task = str(params.get("task", "classification"))
    input_dim = params.get("input_dim")
    n_classes_param = params.get("n_classes")
    auto_train = bool(params.get("auto_train", False))

    if input_dim is None:
        input_dim = min(ctx.expression.shape[0], 500)
    if n_classes_param is None:
        n_classes_param = len(ctx.groups.unique())

    from src.tools.architecture_catalog import get_catalog_summary, CATALOG_BY_NAME

    catalog_summary = get_catalog_summary()

    extraction_prompt = f"""Analyze this paper and map its architecture to a buildable model.

Paper text:
{paper_text[:3000]}

Available architecture patterns:
{catalog_summary}

Available architecture names: {list(CATALOG_BY_NAME.keys())}

Target task: {task}
Input dimension: {input_dim}
Number of classes: {n_classes_param}

Return JSON with:
{{
  "architecture_type": "<name from available architectures>",
  "config": {{
    "input_dim": {input_dim},
    "n_classes": {n_classes_param},
    "hidden_dim": <appropriate size>,
    "n_layers": <number>,
    "n_heads": <if attention-based>,
    "dropout": <0.0-0.5>
  }},
  "rationale": "<why this architecture matches the paper>",
  "key_innovation": "<what's novel in the paper>"
}}

Pick the architecture that best matches the paper's methodology.
Return strictly valid JSON only."""

    design: dict[str, Any] | None = None
    try:
        from src.config import create_llm_backend
        backend = create_llm_backend()
        response = backend.generate(
            extraction_prompt,
            system="You are an AI architecture designer. Map paper methods to buildable architectures.",
            temperature=0.2,
        )
        from src.utils.json_extract import extract_json_object
        design = extract_json_object(response)
    except Exception as e:
        logger.warning("LLM extraction failed (%s), using heuristic matching", e)

    if design is None:
        text_lower = paper_text.lower()
        if any(kw in text_lower for kw in ["attention", "transformer", "self-attention"]):
            arch_type = "gene_transformer"
        elif any(kw in text_lower for kw in ["graph", "gnn", "gcn", "gat", "message passing"]):
            arch_type = "gcn_message_passing"
        elif any(kw in text_lower for kw in ["contrastive", "simclr", "self-supervised"]):
            arch_type = "contrastive_encoder"
        elif any(kw in text_lower for kw in ["vae", "variational", "autoencoder"]):
            arch_type = "expression_vae"
        elif any(kw in text_lower for kw in ["residual", "skip connection", "resnet"]):
            arch_type = "residual_mlp"
        elif any(kw in text_lower for kw in ["multi-modal", "multimodal", "fusion", "cross-attention"]):
            arch_type = "multi_modal_encoder"
        else:
            arch_type = "residual_mlp"

        design = {
            "architecture_type": arch_type,
            "config": {"input_dim": input_dim, "n_classes": n_classes_param,
                       "hidden_dim": 128, "n_layers": 3, "dropout": 0.1},
            "rationale": f"Heuristic match based on keywords in paper text",
            "key_innovation": "Extracted via keyword matching (LLM unavailable)",
        }

    arch_type = str(design.get("architecture_type", "residual_mlp"))
    config = dict(design.get("config", {}))
    config.setdefault("input_dim", input_dim)
    config.setdefault("n_classes", n_classes_param)

    build_result = build_architecture(ctx, {
        "architecture_type": arch_type,
        "config": config,
        "description": f"Designed from paper: {design.get('rationale', '')}",
    })

    if "error" in build_result:
        return build_result

    result = {
        **build_result,
        "design_rationale": design.get("rationale", ""),
        "key_innovation": design.get("key_innovation", ""),
        "paper_architecture_type": arch_type,
        "paper_config": config,
    }

    if auto_train and "model_path" in build_result:
        train_result = train_model_pipeline(ctx, {
            "model_path": build_result["model_path"],
            "epochs": 30,
            "patience": 8,
            "hypothesis_id": params.get("hypothesis_id"),
        })
        result["training_result"] = train_result

    _track_model(
        result, "design_from_paper",
        params.get("hypothesis_id"),
        paper_inspiration=paper_text[:200],
    )
    return _to_python(result)


# ---------------------------------------------------------------------------
# Tool 7: benchmark_model
# ---------------------------------------------------------------------------

def benchmark_model(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a trained model with comprehensive metrics, auto-baselines, and significance tests.

    params:
        model_path: str — path to trained .pt model
        baseline_models: list[str] | None — paths to baseline models to compare
        auto_baseline: bool — auto-train LR + RF baselines if no baselines provided (default True)
        sota_records: list[dict] | None — BaselineRecord dicts from literature for comparison
        metrics: list[str] — metrics to compute
            (default ["accuracy", "f1", "auc_roc", "precision", "recall"])
        gene_subset: list[str] | None — must match training features
        n_top_genes: int — fallback (default 500)
        cross_validate: bool — run k-fold CV (default True)
        cv_folds: int — (default 5)
        significance_test: bool — run statistical significance tests (default True)
    """
    import torch
    import torch.nn as nn

    model_path_str = str(params.get("model_path", ""))
    if not model_path_str or not Path(model_path_str).exists():
        return {"error": f"model_path required and must exist: {model_path_str}"}

    baseline_paths = list(params.get("baseline_models", []))
    auto_baseline = bool(params.get("auto_baseline", True))
    sota_records = list(params.get("sota_records", []))
    requested_metrics = list(params.get("metrics", ["accuracy", "f1", "auc_roc", "precision", "recall"]))
    gene_subset = params.get("gene_subset")
    n_top_genes = int(params.get("n_top_genes", 500))
    do_cv = bool(params.get("cross_validate", True))
    cv_folds = int(params.get("cv_folds", 5))
    do_significance = bool(params.get("significance_test", True))

    device = _get_device()

    def _eval_pytorch_model(path: str) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        arch_type = checkpoint.get("architecture_type", "unknown")
        config = checkpoint.get("config", {})
        saved_gene_names = checkpoint.get("gene_names")
        saved_metrics = checkpoint.get("metrics", {})

        genes_to_use = gene_subset or saved_gene_names
        try:
            X, y, gene_names = _extract_features(ctx, genes_to_use, n_top_genes)
        except Exception as e:
            return {"model_path": path, "error": f"Feature extraction failed: {e}"}

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        from src.tools.architecture_catalog import build_from_catalog, CATALOG_BY_NAME
        if arch_type not in CATALOG_BY_NAME:
            return {
                "model_path": path,
                "architecture": arch_type,
                "saved_metrics": saved_metrics,
                "note": "Cannot rebuild architecture for evaluation",
            }

        config["input_dim"] = X.shape[1]
        n_classes = len(np.unique(y))
        config["n_classes"] = n_classes
        model = build_from_catalog(arch_type, config).to(device)

        try:
            model.load_state_dict(checkpoint["state_dict"])
        except Exception:
            return {
                "model_path": path,
                "architecture": arch_type,
                "saved_metrics": saved_metrics,
                "note": "State dict mismatch — input_dim may have changed",
            }

        model.eval()
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = model(X_t)
            preds = logits.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
        )
        metric_fns = {
            "accuracy": lambda: float(accuracy_score(y, preds)),
            "f1": lambda: float(f1_score(y, preds, average="weighted")),
            "precision": lambda: float(precision_score(y, preds, average="weighted", zero_division=0)),
            "recall": lambda: float(recall_score(y, preds, average="weighted", zero_division=0)),
            "auc_roc": lambda: float(roc_auc_score(y, probs[:, 1])) if n_classes == 2 else None,
        }

        results: dict[str, Any] = {"model_path": path, "architecture": arch_type}
        for m in requested_metrics:
            if m in metric_fns:
                try:
                    results[m] = metric_fns[m]()
                except Exception as e:
                    results[m] = f"error: {e}"

        if do_cv:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_accs: list[float] = []
            for train_idx, test_idx in skf.split(X_scaled, y):
                X_tr = torch.tensor(X_scaled[train_idx], dtype=torch.float32, device=device)
                y_tr = torch.tensor(y[train_idx], dtype=torch.long, device=device)
                X_te_fold = torch.tensor(X_scaled[test_idx], dtype=torch.float32, device=device)

                fold_model = build_from_catalog(arch_type, config).to(device)
                fold_opt = torch.optim.Adam(fold_model.parameters(), lr=1e-3)
                fold_crit = nn.CrossEntropyLoss()
                fold_model.train()
                for _ in range(20):
                    fold_opt.zero_grad()
                    fold_crit(fold_model(X_tr), y_tr).backward()
                    fold_opt.step()
                fold_model.eval()
                with torch.no_grad():
                    fold_preds = fold_model(X_te_fold).argmax(dim=1).cpu().numpy()
                cv_accs.append(float(accuracy_score(y[test_idx], fold_preds)))

            results["cv_scores"] = cv_accs
            results["cv_mean"] = float(np.mean(cv_accs))
            results["cv_std"] = float(np.std(cv_accs))

        results["n_samples"] = X.shape[0]
        results["n_features"] = X.shape[1]
        results["predictions"] = preds.tolist()
        return results

    primary = _eval_pytorch_model(model_path_str)

    comparisons: list[dict[str, Any]] = []
    for bp in baseline_paths:
        if Path(bp).exists():
            comparisons.append(_eval_pytorch_model(bp))

    # Auto-baseline: train LR + RF if no explicit baselines provided
    auto_baselines: list[dict[str, Any]] = []
    if auto_baseline and not baseline_paths:
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score as acc_score, f1_score as f1_fn
        from sklearn.model_selection import cross_val_score

        try:
            X, y, gn = _extract_features(ctx, gene_subset, n_top_genes)
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)

            for name, clf in [("Logistic Regression", LogisticRegression(max_iter=1000)),
                              ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42))]:
                clf.fit(X_s, y)
                preds = clf.predict(X_s)
                cv = cross_val_score(clf, X_s, y, cv=min(cv_folds, len(y) // 2), scoring="accuracy")
                bl_entry = {
                    "name": name,
                    "source": "auto_baseline",
                    "accuracy": float(acc_score(y, preds)),
                    "f1": float(f1_fn(y, preds, average="weighted")),
                    "cv_mean": float(cv.mean()),
                    "cv_std": float(cv.std()),
                    "predictions": preds.tolist(),
                }
                auto_baselines.append(bl_entry)
                print(f"  Auto-baseline {name}: accuracy={bl_entry['accuracy']:.4f}, cv={bl_entry['cv_mean']:.4f}±{bl_entry['cv_std']:.4f}")
        except Exception as e:
            print(f"  Auto-baseline failed: {e}")

    # Statistical significance tests
    significance_results: list[dict[str, Any]] = []
    if do_significance and primary.get("predictions"):
        primary_preds = np.array(primary["predictions"])
        all_comparison_preds = []

        for bl in auto_baselines:
            if bl.get("predictions"):
                all_comparison_preds.append((bl["name"], np.array(bl["predictions"])))
        for comp in comparisons:
            if comp.get("predictions"):
                all_comparison_preds.append((comp.get("architecture", "baseline"), np.array(comp["predictions"])))

        for bl_name, bl_preds in all_comparison_preds:
            try:
                X_check, y_check, _ = _extract_features(ctx, gene_subset, n_top_genes)
                y_true = y_check

                primary_correct = (primary_preds == y_true).astype(int)
                bl_correct = (bl_preds == y_true).astype(int)

                # McNemar's test
                b = int(np.sum((primary_correct == 1) & (bl_correct == 0)))
                c = int(np.sum((primary_correct == 0) & (bl_correct == 1)))

                if b + c > 0:
                    from scipy.stats import binom_test
                    try:
                        p_value = float(binom_test(b, b + c, 0.5))
                    except Exception:
                        chi2 = (abs(b - c) - 1) ** 2 / max(b + c, 1)
                        from scipy.stats import chi2 as chi2_dist
                        p_value = float(1.0 - chi2_dist.cdf(chi2, df=1))
                else:
                    p_value = 1.0

                significance_results.append({
                    "novel_vs": bl_name,
                    "novel_accuracy": primary.get("accuracy"),
                    "baseline_accuracy": next(
                        (bl["accuracy"] for bl in auto_baselines if bl["name"] == bl_name), None
                    ) or next(
                        (c.get("accuracy") for c in comparisons if c.get("architecture") == bl_name), None
                    ),
                    "mcnemar_p_value": round(p_value, 6),
                    "significant_at_0.05": p_value < 0.05,
                    "b_novel_right_bl_wrong": b,
                    "c_novel_wrong_bl_right": c,
                })
            except Exception as e:
                significance_results.append({"novel_vs": bl_name, "error": str(e)})

    # Add SOTA literature records for comparison table
    sota_comparison: list[dict[str, Any]] = []
    for sr in sota_records:
        sota_comparison.append({
            "name": sr.get("name", "SOTA"),
            "source": "literature",
            "metrics": sr.get("metrics", {}),
            "paper": sr.get("paper", ""),
        })

    # Clean predictions from results before returning (large)
    for d in [primary] + comparisons + auto_baselines:
        d.pop("predictions", None)

    arch_hash = hashlib.sha256(model_path_str.encode()).hexdigest()[:12]

    result = _to_python({
        "model_id": _model_id(),
        "primary_evaluation": primary,
        "baseline_comparisons": comparisons,
        "auto_baselines": auto_baselines,
        "sota_literature": sota_comparison,
        "significance_tests": significance_results,
        "arch_hash": arch_hash,
        "metrics_computed": requested_metrics,
    })

    _track_model(
        {**result, **(primary if isinstance(primary, dict) else {})},
        "benchmark",
        params.get("hypothesis_id"),
    )
    return result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_model_builder_tools(registry: dict[str, Callable]) -> None:
    """Register all model-builder tools into the TOOL_REGISTRY."""

    tools: dict[str, Callable] = {
        "build_architecture": build_architecture,
        "train_model_pipeline": train_model_pipeline,
        "finetune_protein_lm": finetune_protein_lm,
        "finetune_genomic_lm": finetune_genomic_lm,
        "build_graph_model": build_graph_model,
        "design_from_paper": design_from_paper,
        "benchmark_model": benchmark_model,
    }

    for name, fn in tools.items():
        def _make_safe(tool_fn: Callable, tool_name: str) -> Callable:
            def safe_wrapper(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
                try:
                    return tool_fn(ctx, params)
                except Exception as e:
                    import traceback
                    return {
                        "error": f"{tool_name} failed: {type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                    }
            safe_wrapper.__name__ = tool_name
            safe_wrapper.__doc__ = tool_fn.__doc__
            return safe_wrapper

        registry[name] = _make_safe(fn, name)

    print(f"Registered {len(tools)} model-builder tools: {', '.join(tools)}")
