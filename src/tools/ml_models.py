"""ML model-building tools for the research agent.

Lets the agent autonomously train, evaluate, and iterate on predictive models
when hypotheses require them. All tools follow the TOOL_REGISTRY signature:
``(ctx: ToolContext, params: dict) -> dict``.

Heavy imports (torch, sklearn estimators) are deferred to call-time so the
module loads quickly.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.tools.data_analysis import ToolContext

_MODEL_STORE = None


def _get_model_store():
    """Lazy-init a global ModelStore so every trained model is tracked."""
    global _MODEL_STORE
    if _MODEL_STORE is None:
        from src.memory.model_store import ModelStore
        _MODEL_STORE = ModelStore(Path("./outputs/model_registry.json"))
    return _MODEL_STORE


def _track_model(result: dict[str, Any], task: str, hypothesis_id: str | None = None) -> None:
    """Persist a trained model record to ModelStore."""
    try:
        from src.memory.model_store import ModelRecord
        store = _get_model_store()
        store.add(ModelRecord(
            id=result.get("model_id", _model_id()),
            model_type=result.get("model_type", result.get("architecture", "unknown")),
            task=task,
            hypothesis_id=hypothesis_id,
            experiment_id=None,
            metrics={k: v for k, v in result.items()
                     if k in ("accuracy", "f1", "auc_roc", "cross_val_mean",
                              "final_reconstruction_error")
                     and isinstance(v, (int, float))},
            hyperparameters={k: v for k, v in result.items()
                            if k in ("n_features", "n_samples", "embedding_dim",
                                     "latent_dim", "method")},
            feature_genes=result.get("gene_set", result.get("selected_genes", [])),
            model_path=result.get("model_path", ""),
            training_time_s=result.get("training_time_s", 0.0),
            notes=task,
        ))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_id() -> str:
    """Generate a short unique model identifier."""
    return "model_" + hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]


def _ensure_output(ctx: ToolContext) -> Path:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    return ctx.output_dir


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
    n_top_genes: int = 100,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract (X, y, gene_names) from the ToolContext.

    Expression is genes-by-samples; we transpose to samples-by-genes.
    """
    expr = ctx.expression

    if gene_subset:
        available = [g for g in gene_subset if g in expr.index]
        if not available:
            raise ValueError(
                f"None of the specified genes found. Available (first 10): "
                f"{list(expr.index[:10])}"
            )
        expr = expr.loc[available]
    else:
        var = expr.var(axis=1).sort_values(ascending=False)
        top = var.head(n_top_genes).index
        expr = expr.loc[top]

    X = expr.T.astype(float).values
    gene_names = list(expr.index.astype(str))

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(ctx.groups.values)

    return X, y, gene_names


def _get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Tool: train_classifier
# ---------------------------------------------------------------------------

def train_classifier(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Train a sample classifier on expression data.

    params:
        model_type: str — "logistic_regression", "random_forest", "svm",
                          "gradient_boosting", "mlp"
        gene_subset: list[str] | None — genes to use as features
        n_top_genes: int — top N by variance if gene_subset is None (default 100)
        test_fraction: float — held-out test fraction (default 0.3)
        cv_folds: int — cross-validation folds (default 5)
        hyperparams: dict | None — model-specific hyperparameters
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    model_type = str(params.get("model_type", "logistic_regression"))
    gene_subset = params.get("gene_subset")
    n_top_genes = int(params.get("n_top_genes", 100))
    test_fraction = float(params.get("test_fraction", 0.3))
    cv_folds = int(params.get("cv_folds", 5))
    hp = params.get("hyperparams") or {}

    X, y, gene_names = _extract_features(ctx, gene_subset, n_top_genes)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_fraction, random_state=42, stratify=y,
    )

    print(f"Training {model_type} on {X.shape[0]} samples, {X.shape[1]} features")

    t0 = time.time()

    if model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=int(hp.get("max_iter", 1000)),
            C=float(hp.get("C", 1.0)),
            random_state=42,
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(hp.get("n_estimators", 200)),
            max_depth=hp.get("max_depth"),
            random_state=42,
        )
    elif model_type == "svm":
        model = SVC(
            C=float(hp.get("C", 1.0)),
            kernel=str(hp.get("kernel", "rbf")),
            probability=True,
            random_state=42,
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=int(hp.get("n_estimators", 200)),
            learning_rate=float(hp.get("learning_rate", 0.1)),
            max_depth=int(hp.get("max_depth", 3)),
            random_state=42,
        )
    elif model_type == "mlp":
        return _train_mlp_classifier(
            ctx, X_train, X_test, y_train, y_test, X_scaled, y,
            gene_names, cv_folds, hp,
        )
    else:
        return {"error": f"Unknown model_type: {model_type}. "
                f"Choose from: logistic_regression, random_forest, svm, "
                f"gradient_boosting, mlp"}

    model.fit(X_train, y_train)
    training_time = time.time() - t0

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None

    cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring="accuracy")

    # Feature importances
    top_features = _sklearn_feature_importance(model, model_type, gene_names)

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    mid = _model_id()
    out_dir = _ensure_output(ctx)
    model_path = out_dir / f"{mid}.pkl"
    import joblib
    joblib.dump({"model": model, "scaler": scaler, "gene_names": gene_names}, model_path)

    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc} | CV mean: {cv_scores.mean():.4f}")
    print(f"Model saved to {model_path}")

    result = _to_python({
        "model_id": mid,
        "model_type": model_type,
        "accuracy": acc,
        "f1": f1,
        "auc_roc": auc,
        "cross_val_scores": cv_scores.tolist(),
        "cross_val_mean": float(cv_scores.mean()),
        "cross_val_std": float(cv_scores.std()),
        "top_features": top_features[:20],
        "confusion_matrix": cm.tolist(),
        "classification_report": cr,
        "model_path": str(model_path),
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "training_time_s": training_time,
    })
    _track_model(result, "classification", params.get("hypothesis_id"))
    return result


def _sklearn_feature_importance(
    model: Any, model_type: str, gene_names: list[str],
) -> list[dict[str, Any]]:
    """Extract feature importances from an sklearn estimator."""
    if model_type in ("random_forest", "gradient_boosting"):
        imp = model.feature_importances_
    elif model_type == "logistic_regression":
        imp = np.abs(model.coef_).flatten()
    elif model_type == "svm" and model.kernel == "linear":
        imp = np.abs(model.coef_).flatten()
    else:
        return []

    order = np.argsort(-imp)
    return [
        {"gene": gene_names[i], "importance": float(imp[i])}
        for i in order
    ]


def _train_mlp_classifier(
    ctx: ToolContext,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_all: np.ndarray,
    y_all: np.ndarray,
    gene_names: list[str],
    cv_folds: int,
    hp: dict,
) -> dict[str, Any]:
    """Train a PyTorch MLP classifier (GPU-accelerated)."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )
    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPClassifier

    device = _get_device()
    print(f"Training MLP on {device}")

    hidden_dims = hp.get("hidden_dims", [128, 64])
    epochs = int(hp.get("epochs", 50))
    lr = float(hp.get("learning_rate", 1e-3))
    batch_size = int(hp.get("batch_size", 32))

    n_classes = len(np.unique(y_all))
    input_dim = X_train.shape[1]

    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.3)])
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    net = nn.Sequential(*layers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_te_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    t0 = time.time()
    loss_history: list[float] = []

    for epoch in range(epochs):
        net.train()
        indices = torch.randperm(len(X_tr_t), device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(X_tr_t), batch_size):
            idx = indices[start : start + batch_size]
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            optimizer.zero_grad()
            out = net(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

    training_time = time.time() - t0

    net.eval()
    with torch.no_grad():
        logits = net(X_te_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = float(roc_auc_score(y_test, probs[:, 1])) if n_classes == 2 else None
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    # sklearn MLP for cross-val (lighter weight)
    sk_mlp = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_dims), max_iter=epochs, random_state=42,
    )
    cv_scores = cross_val_score(sk_mlp, X_all, y_all, cv=cv_folds, scoring="accuracy")

    mid = _model_id()
    out_dir = _ensure_output(ctx)
    model_path = out_dir / f"{mid}_mlp.pt"
    torch.save({
        "state_dict": net.state_dict(),
        "architecture": {"input_dim": input_dim, "hidden_dims": hidden_dims, "n_classes": n_classes},
        "gene_names": gene_names,
    }, model_path)

    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc} | CV mean: {cv_scores.mean():.4f}")

    result = _to_python({
        "model_id": mid,
        "model_type": "mlp",
        "accuracy": acc,
        "f1": f1,
        "auc_roc": auc,
        "cross_val_scores": cv_scores.tolist(),
        "cross_val_mean": float(cv_scores.mean()),
        "cross_val_std": float(cv_scores.std()),
        "top_features": [],
        "confusion_matrix": cm.tolist(),
        "classification_report": cr,
        "training_loss_history": loss_history,
        "model_path": str(model_path),
        "n_samples": X_train.shape[0] + X_test.shape[0],
        "n_features": X_train.shape[1],
        "training_time_s": training_time,
        "device": str(device),
    })
    _track_model(result, "classification")
    return result


# ---------------------------------------------------------------------------
# Tool: train_neural_network
# ---------------------------------------------------------------------------

def train_neural_network(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Train a PyTorch neural network on expression data (GPU-accelerated).

    params:
        architecture: str — "mlp", "autoencoder", "vae"
        hidden_dims: list[int] — hidden layer sizes (default [256, 128, 64])
        task: str — "classification", "reconstruction", "embedding"
        epochs: int — training epochs (default 50)
        learning_rate: float — (default 1e-3)
        batch_size: int — (default 32)
        gene_subset: list[str] | None — features to use
        n_top_genes: int — fallback feature count (default 500)
    """
    import torch
    import torch.nn as nn

    architecture = str(params.get("architecture", "autoencoder"))
    hidden_dims = list(params.get("hidden_dims", [256, 128, 64]))
    task = str(params.get("task", "reconstruction" if architecture != "mlp" else "classification"))
    epochs = int(params.get("epochs", 50))
    lr = float(params.get("learning_rate", 1e-3))
    batch_size = int(params.get("batch_size", 32))
    gene_subset = params.get("gene_subset")
    n_top_genes = int(params.get("n_top_genes", 500))

    X, y, gene_names = _extract_features(ctx, gene_subset, n_top_genes)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    device = _get_device()
    print(f"Training {architecture} ({task}) on {device} — "
          f"{X.shape[0]} samples, {X.shape[1]} features, {epochs} epochs")

    t0 = time.time()

    if architecture == "mlp" and task == "classification":
        return _train_mlp_classifier(
            ctx,
            *_split_data(X_scaled, y),
            X_scaled, y, gene_names,
            cv_folds=5,
            hp={"hidden_dims": hidden_dims, "epochs": epochs,
                "learning_rate": lr, "batch_size": batch_size},
        )
    elif architecture == "autoencoder":
        result = _train_autoencoder(
            X_scaled, gene_names, hidden_dims, epochs, lr, batch_size, device,
        )
    elif architecture == "vae":
        result = _train_vae(
            X_scaled, gene_names, hidden_dims, epochs, lr, batch_size, device,
        )
    else:
        return {"error": f"Unknown architecture: {architecture}. Choose: mlp, autoencoder, vae"}

    training_time = time.time() - t0
    result["training_time_s"] = training_time

    mid = _model_id()
    result["model_id"] = mid
    out_dir = _ensure_output(ctx)

    model_path = out_dir / f"{mid}_{architecture}.pt"
    torch.save(result.pop("_state"), model_path)
    result["model_path"] = str(model_path)

    if "embeddings" in result:
        emb_df = pd.DataFrame(
            result["embeddings"],
            index=[str(s) for s in ctx.expression.columns],
        )
        emb_path = out_dir / f"{mid}_embeddings.csv"
        emb_df.to_csv(emb_path)
        result["embeddings_path"] = str(emb_path)
        result["embeddings"] = result["embeddings"].tolist() if hasattr(result["embeddings"], "tolist") else result["embeddings"]

    print(f"Training complete in {training_time:.1f}s — saved to {model_path}")
    result = _to_python(result)
    _track_model(result, result.get("task", "neural_network"), params.get("hypothesis_id"))
    return result


def _split_data(X: np.ndarray, y: np.ndarray):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def _train_autoencoder(
    X: np.ndarray,
    gene_names: list[str],
    hidden_dims: list[int],
    epochs: int,
    lr: float,
    batch_size: int,
    device: Any,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    input_dim = X.shape[1]

    # Encoder
    enc_layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        enc_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
        prev = h
    encoder = nn.Sequential(*enc_layers)

    # Decoder (mirror)
    dec_layers: list[nn.Module] = []
    rev = list(reversed(hidden_dims))
    prev = rev[0]
    for h in rev[1:]:
        dec_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
        prev = h
    dec_layers.append(nn.Linear(prev, input_dim))
    decoder = nn.Sequential(*dec_layers)

    model = nn.Sequential(encoder, decoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    loss_history: list[float] = []
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_t), device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(X_t), batch_size):
            idx = indices[start : start + batch_size]
            xb = X_t[idx]
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}  recon_loss={avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        embeddings = encoder(X_t).cpu().numpy()
        recon = model(X_t)
        recon_error = criterion(recon, X_t).item()

    return {
        "architecture": "autoencoder",
        "task": "reconstruction",
        "training_loss_history": loss_history,
        "final_reconstruction_error": recon_error,
        "latent_dim": hidden_dims[-1],
        "embeddings": embeddings,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "device": str(device),
        "_state": {
            "state_dict": model.state_dict(),
            "architecture_config": {
                "input_dim": input_dim,
                "hidden_dims": hidden_dims,
            },
            "gene_names": gene_names,
        },
    }


def _train_vae(
    X: np.ndarray,
    gene_names: list[str],
    hidden_dims: list[int],
    epochs: int,
    lr: float,
    batch_size: int,
    device: Any,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    input_dim = X.shape[1]
    latent_dim = hidden_dims[-1]

    class VAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            enc: list[nn.Module] = []
            prev = input_dim
            for h in hidden_dims[:-1]:
                enc.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
                prev = h
            self.encoder = nn.Sequential(*enc)
            self.fc_mu = nn.Linear(prev, latent_dim)
            self.fc_logvar = nn.Linear(prev, latent_dim)

            dec: list[nn.Module] = []
            rev = list(reversed(hidden_dims))
            prev = rev[0]
            for h in rev[1:]:
                dec.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
                prev = h
            dec.append(nn.Linear(prev, input_dim))
            self.decoder = nn.Sequential(*dec)

        def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return self.decoder(z)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    def vae_loss(recon: torch.Tensor, x: torch.Tensor,
                 mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl

    loss_history: list[float] = []
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_t), device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(X_t), batch_size):
            idx = indices[start : start + batch_size]
            xb = X_t[idx]
            optimizer.zero_grad()
            recon, mu, logvar = model(xb)
            loss = vae_loss(recon, xb, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / (max(n_batches, 1) * batch_size)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}  vae_loss={avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(X_t)
        embeddings = mu.cpu().numpy()
        recon, _, _ = model(X_t)
        recon_error = nn.functional.mse_loss(recon, X_t).item()

    return {
        "architecture": "vae",
        "task": "reconstruction",
        "training_loss_history": loss_history,
        "final_reconstruction_error": recon_error,
        "latent_dim": latent_dim,
        "embeddings": embeddings,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "device": str(device),
        "_state": {
            "state_dict": model.state_dict(),
            "architecture_config": {
                "input_dim": input_dim,
                "hidden_dims": hidden_dims,
                "latent_dim": latent_dim,
            },
            "gene_names": gene_names,
        },
    }


# ---------------------------------------------------------------------------
# Tool: evaluate_model
# ---------------------------------------------------------------------------

def evaluate_model(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a trained model with additional metrics or on held-out data.

    params:
        model_path: str — path to saved model (.pkl for sklearn, .pt for PyTorch)
        metrics: list[str] — subset of ["accuracy","f1","auc_roc","precision","recall","mcc"]
        gene_subset: list[str] | None — must match training features
        cross_validate: bool — run full CV evaluation (default False)
    """
    model_path = str(params.get("model_path", ""))
    if not model_path:
        return {"error": "model_path is required"}
    path = Path(model_path)
    if not path.exists():
        return {"error": f"Model file not found: {model_path}"}

    requested_metrics = list(params.get("metrics", ["accuracy", "f1", "auc_roc"]))
    gene_subset = params.get("gene_subset")
    do_cv = bool(params.get("cross_validate", False))

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if path.suffix == ".pkl":
        import joblib
        bundle = joblib.load(path)
        model = bundle["model"]
        gene_names = bundle.get("gene_names", [])
        scaler = bundle.get("scaler")

        X, y, _ = _extract_features(ctx, gene_subset or gene_names, len(gene_names))
        if scaler is not None:
            X = scaler.transform(X)

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

        results: dict[str, Any] = {}
        metric_fn = {
            "accuracy": lambda: accuracy_score(y, y_pred),
            "f1": lambda: f1_score(y, y_pred, average="weighted"),
            "auc_roc": lambda: float(roc_auc_score(y, y_prob)) if y_prob is not None else None,
            "precision": lambda: precision_score(y, y_pred, average="weighted"),
            "recall": lambda: recall_score(y, y_pred, average="weighted"),
            "mcc": lambda: matthews_corrcoef(y, y_pred),
        }
        for m in requested_metrics:
            if m in metric_fn:
                try:
                    results[m] = float(metric_fn[m]()) if metric_fn[m]() is not None else None
                except Exception as e:
                    results[m] = f"error: {e}"

        if do_cv:
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
            results["cross_val_scores"] = cv_scores.tolist()
            results["cross_val_mean"] = float(cv_scores.mean())

        print(f"Evaluation on {X.shape[0]} samples: {results}")
        return _to_python({"model_path": model_path, "metrics": results, "n_samples": X.shape[0]})

    elif path.suffix == ".pt":
        import torch
        import torch.nn as nn
        device = _get_device()
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        arch = checkpoint.get("architecture", checkpoint.get("architecture_config", {}))
        gene_names_saved = checkpoint.get("gene_names", [])

        X, y, _ = _extract_features(ctx, gene_subset or gene_names_saved, len(gene_names_saved))
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)

        if "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
            recon_test = any("decoder" in k for k in sd)
            if recon_test:
                recon_err = float(nn.functional.mse_loss(X_t, X_t).item())
                return _to_python({
                    "model_path": model_path,
                    "note": "Autoencoder/VAE evaluation requires architecture rebuild; "
                            "returning data-level stats",
                    "n_samples": X.shape[0],
                    "n_features": X.shape[1],
                })

        return _to_python({
            "model_path": model_path,
            "note": "PyTorch model loaded; detailed evaluation requires matching architecture",
            "n_samples": X.shape[0],
        })

    return {"error": f"Unsupported model format: {path.suffix}"}


# ---------------------------------------------------------------------------
# Tool: feature_selection
# ---------------------------------------------------------------------------

def feature_selection(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """ML-based feature selection to identify discriminative genes.

    params:
        method: str — "lasso", "random_forest", "mutual_information",
                      "boruta", "recursive_elimination"
        n_features: int — target number of features (default 50)
        gene_subset: list[str] | None — restrict to these genes
        n_top_genes: int — initial pool size (default 2000)
    """
    method = str(params.get("method", "random_forest"))
    n_features = int(params.get("n_features", 50))
    gene_subset = params.get("gene_subset")
    n_top_genes = int(params.get("n_top_genes", 2000))

    X, y, gene_names = _extract_features(ctx, gene_subset, n_top_genes)

    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    print(f"Feature selection: {method} — selecting {n_features} from {len(gene_names)} genes")

    t0 = time.time()

    if method == "lasso":
        from sklearn.linear_model import LassoCV
        lasso = LassoCV(cv=5, random_state=42, max_iter=2000).fit(X_scaled, y)
        imp = np.abs(lasso.coef_)
        details = {"alpha": float(lasso.alpha_)}

    elif method == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)
        imp = rf.feature_importances_
        details = {"n_estimators": 200}

    elif method == "mutual_information":
        from sklearn.feature_selection import mutual_info_classif
        imp = mutual_info_classif(X_scaled, y, random_state=42)
        details = {}

    elif method == "recursive_elimination":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import RFECV
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rfe = RFECV(rf, step=max(1, len(gene_names) // 20), cv=3, scoring="accuracy", n_jobs=-1)
        rfe.fit(X_scaled, y)
        imp = np.zeros(len(gene_names))
        imp[rfe.support_] = 1.0 / rfe.ranking_[rfe.support_]
        details = {"optimal_n_features": int(rfe.n_features_)}

    elif method == "boruta":
        # Boruta-style: compare real features against shadow (shuffled) features
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rng = np.random.RandomState(42)
        shadow = X_scaled[:, rng.permutation(X_scaled.shape[1])]
        X_aug = np.hstack([X_scaled, shadow])
        rf.fit(X_aug, y)
        real_imp = rf.feature_importances_[:X_scaled.shape[1]]
        shadow_imp = rf.feature_importances_[X_scaled.shape[1]:]
        threshold = np.max(shadow_imp)
        imp = np.where(real_imp > threshold, real_imp, 0.0)
        details = {"shadow_threshold": float(threshold)}

    else:
        return {"error": f"Unknown method: {method}. Choose: lasso, random_forest, "
                f"mutual_information, boruta, recursive_elimination"}

    elapsed = time.time() - t0
    order = np.argsort(-imp)
    top_idx = order[:n_features]

    selected = [
        {"gene": gene_names[i], "importance": float(imp[i])}
        for i in top_idx if imp[i] > 0
    ]
    selected_genes = [s["gene"] for s in selected]

    print(f"Selected {len(selected)} genes in {elapsed:.1f}s — "
          f"top: {', '.join(selected_genes[:5])}")

    return _to_python({
        "method": method,
        "selected_genes": selected_genes,
        "importance_scores": selected,
        "n_requested": n_features,
        "n_selected": len(selected),
        "method_details": details,
        "elapsed_s": elapsed,
    })


# ---------------------------------------------------------------------------
# Tool: train_gene_embeddings
# ---------------------------------------------------------------------------

def train_gene_embeddings(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Learn gene embeddings from expression patterns using a neural network.

    params:
        embedding_dim: int — dimension of embeddings (default 64)
        method: str — "autoencoder", "gene2vec_style", "contrastive"
        epochs: int — (default 30)
        gene_subset: list[str] | None — restrict to these genes
        n_top_genes: int — fallback pool size (default 2000)
    """
    import torch
    import torch.nn as nn

    embedding_dim = int(params.get("embedding_dim", 64))
    method = str(params.get("method", "autoencoder"))
    epochs = int(params.get("epochs", 30))
    gene_subset = params.get("gene_subset")
    n_top_genes = int(params.get("n_top_genes", 2000))
    lr = float(params.get("learning_rate", 1e-3))
    batch_size = int(params.get("batch_size", 32))

    # Gene embeddings: each gene is a data point with samples as features
    expr = ctx.expression
    if gene_subset:
        available = [g for g in gene_subset if g in expr.index]
        if not available:
            return {"error": "None of the specified genes found in expression data"}
        expr = expr.loc[available]
    else:
        var = expr.var(axis=1).sort_values(ascending=False)
        expr = expr.loc[var.head(n_top_genes).index]

    from sklearn.preprocessing import StandardScaler
    X_genes = StandardScaler().fit_transform(expr.astype(float).values)
    gene_names = list(expr.index.astype(str))
    n_genes, n_samples = X_genes.shape

    device = _get_device()
    print(f"Learning {embedding_dim}-dim embeddings for {n_genes} genes on {device} ({method})")

    t0 = time.time()

    if method in ("autoencoder", "gene2vec_style"):
        enc_layers: list[nn.Module] = []
        dims = [n_samples, max(n_samples // 2, embedding_dim * 2), embedding_dim]
        for i in range(len(dims) - 1):
            enc_layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        rev_dims = list(reversed(dims))
        for i in range(len(rev_dims) - 1):
            dec_layers.extend([nn.Linear(rev_dims[i], rev_dims[i + 1]), nn.ReLU()])
        decoder = nn.Sequential(*dec_layers)

        ae_model = nn.Sequential(encoder, decoder).to(device)
        optimizer = torch.optim.Adam(ae_model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_genes, dtype=torch.float32, device=device)

        loss_history: list[float] = []
        for epoch in range(epochs):
            ae_model.train()
            indices = torch.randperm(n_genes, device=device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_genes, batch_size):
                idx = indices[start : start + batch_size]
                xb = X_t[idx]
                optimizer.zero_grad()
                recon = ae_model(xb)
                loss = criterion(recon, xb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(avg_loss)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

        ae_model.eval()
        with torch.no_grad():
            embeddings = encoder(X_t).cpu().numpy()

    elif method == "contrastive":
        projector = nn.Sequential(
            nn.Linear(n_samples, max(n_samples // 2, embedding_dim * 2)),
            nn.ReLU(),
            nn.Linear(max(n_samples // 2, embedding_dim * 2), embedding_dim),
        ).to(device)

        optimizer = torch.optim.Adam(projector.parameters(), lr=lr)
        X_t = torch.tensor(X_genes, dtype=torch.float32, device=device)
        temperature = 0.1

        loss_history = []
        for epoch in range(epochs):
            projector.train()
            # Augment: add noise for positive pairs
            noise = torch.randn_like(X_t) * 0.1
            z1 = nn.functional.normalize(projector(X_t), dim=1)
            z2 = nn.functional.normalize(projector(X_t + noise), dim=1)

            sim = torch.mm(z1, z2.T) / temperature
            labels = torch.arange(n_genes, device=device)
            loss = nn.functional.cross_entropy(sim, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs}  contrastive_loss={loss.item():.4f}")

        projector.eval()
        with torch.no_grad():
            embeddings = nn.functional.normalize(projector(X_t), dim=1).cpu().numpy()
    else:
        return {"error": f"Unknown method: {method}. Choose: autoencoder, gene2vec_style, contrastive"}

    training_time = time.time() - t0

    # Find similar gene pairs by cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    sim_matrix = cos_sim(embeddings)
    np.fill_diagonal(sim_matrix, -1)
    similar_pairs: list[dict[str, Any]] = []
    n_pairs = min(20, n_genes * (n_genes - 1) // 2)
    flat_idx = np.argsort(-sim_matrix.ravel())
    seen: set[tuple[str, str]] = set()
    for fi in flat_idx:
        i, j = divmod(int(fi), n_genes)
        if i >= j:
            continue
        pair = (gene_names[i], gene_names[j])
        if pair not in seen:
            seen.add(pair)
            similar_pairs.append({
                "gene_a": pair[0],
                "gene_b": pair[1],
                "similarity": float(sim_matrix[i, j]),
            })
        if len(similar_pairs) >= n_pairs:
            break

    mid = _model_id()
    out_dir = _ensure_output(ctx)

    emb_df = pd.DataFrame(embeddings, index=gene_names)
    emb_path = out_dir / f"{mid}_gene_embeddings.csv"
    emb_df.to_csv(emb_path)

    emb_dict = {gene_names[i]: embeddings[i].tolist() for i in range(n_genes)}

    print(f"Learned {embedding_dim}-dim embeddings for {n_genes} genes in {training_time:.1f}s")

    result = _to_python({
        "model_id": mid,
        "method": method,
        "embedding_dim": embedding_dim,
        "n_genes": n_genes,
        "embeddings_path": str(emb_path),
        "training_loss": loss_history,
        "similar_gene_pairs": similar_pairs,
        "training_time_s": training_time,
        "device": str(device),
    })
    _track_model(result, "gene_embeddings", params.get("hypothesis_id"))
    return result


# ---------------------------------------------------------------------------
# Tool: cross_validate_hypothesis
# ---------------------------------------------------------------------------

def cross_validate_hypothesis(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Rigorously test if a gene set can distinguish groups using multiple ML methods.

    params:
        gene_set: list[str] — genes to test
        methods: list[str] — classifiers to use (default all)
        n_permutations: int — permutation tests for p-value (default 100)
        cv_folds: int — cross-validation folds (default 5)
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    gene_set = list(params.get("gene_set", []))
    if not gene_set:
        return {"error": "gene_set is required — provide a list of gene names to test"}

    methods = list(params.get("methods", [
        "logistic_regression", "random_forest", "svm", "gradient_boosting",
    ]))
    n_permutations = int(params.get("n_permutations", 100))
    cv_folds = int(params.get("cv_folds", 5))

    X, y, gene_names = _extract_features(ctx, gene_set, len(gene_set))
    X_scaled = StandardScaler().fit_transform(X)

    print(f"Cross-validating hypothesis with {len(gene_names)} genes, "
          f"{len(methods)} methods, {n_permutations} permutations")

    estimators = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "svm": SVC(kernel="rbf", probability=True, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    method_results: list[dict[str, Any]] = []

    for method_name in methods:
        if method_name not in estimators:
            method_results.append({"method": method_name, "error": "unknown method"})
            continue

        est = estimators[method_name]
        scores = cross_val_score(est, X_scaled, y, cv=cv, scoring="accuracy")

        perm_scores: list[float] = []
        rng = np.random.RandomState(42)
        for p in range(n_permutations):
            y_perm = rng.permutation(y)
            perm_s = cross_val_score(est, X_scaled, y_perm, cv=cv, scoring="accuracy")
            perm_scores.append(float(perm_s.mean()))

        perm_p = float(np.mean(np.array(perm_scores) >= scores.mean()))

        method_results.append({
            "method": method_name,
            "cv_scores": scores.tolist(),
            "cv_mean": float(scores.mean()),
            "cv_std": float(scores.std()),
            "permutation_p_value": perm_p,
            "permutation_mean": float(np.mean(perm_scores)),
        })
        print(f"  {method_name}: CV={scores.mean():.4f}±{scores.std():.4f}, "
              f"perm_p={perm_p:.4f}")

    # Aggregate assessment
    real_means = [r["cv_mean"] for r in method_results if "cv_mean" in r]
    perm_ps = [r["permutation_p_value"] for r in method_results if "permutation_p_value" in r]

    aggregate_cv = float(np.mean(real_means)) if real_means else 0.0
    n_significant = sum(1 for p in perm_ps if p < 0.05)

    if aggregate_cv > 0.8 and n_significant == len(perm_ps):
        assessment = "strong_evidence"
    elif aggregate_cv > 0.7 and n_significant >= len(perm_ps) / 2:
        assessment = "moderate_evidence"
    elif aggregate_cv > 0.6:
        assessment = "weak_evidence"
    else:
        assessment = "insufficient_evidence"

    print(f"Aggregate: CV={aggregate_cv:.4f}, {n_significant}/{len(perm_ps)} "
          f"significant → {assessment}")

    return _to_python({
        "gene_set": gene_names,
        "n_genes": len(gene_names),
        "per_method": method_results,
        "aggregate_cv_accuracy": aggregate_cv,
        "n_methods_significant": n_significant,
        "assessment": assessment,
        "n_permutations": n_permutations,
        "cv_folds": cv_folds,
    })


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_ml_tools(registry: dict[str, Callable]) -> None:
    """Register all ML tools into the TOOL_REGISTRY, defensive against import failures."""

    tools: dict[str, Callable] = {
        "train_classifier": train_classifier,
        "train_neural_network": train_neural_network,
        "evaluate_model": evaluate_model,
        "feature_selection": feature_selection,
        "train_gene_embeddings": train_gene_embeddings,
        "cross_validate_hypothesis": cross_validate_hypothesis,
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

    print(f"Registered {len(tools)} ML tools: {', '.join(tools)}")
