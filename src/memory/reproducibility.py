"""Reproducibility tracking for experiments.

Captures data hashes, random seeds, package versions, and environment
snapshots so that any experiment can be reproduced exactly.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ReproducibilityBundle:
    """Everything needed to reproduce a single experiment run."""

    experiment_id: str
    session_id: str
    cycle: int

    # Data fingerprints
    expression_hash: str
    groups_hash: str
    n_samples: int
    n_genes: int
    sample_ids: list[str]

    # Random seeds
    numpy_seed: int | None
    torch_seed: int | None
    python_hash_seed: str | None

    # Environment
    python_version: str
    platform_info: str
    package_versions: dict[str, str]

    # Config
    config_snapshot: dict[str, Any]
    tool_sequence: list[str]

    created_at: str = field(default_factory=_utc_now)


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Deterministic hash of a DataFrame (content only, order-insensitive)."""
    buf = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(buf.tobytes()).hexdigest()[:16]


def _hash_series(s: pd.Series) -> str:
    buf = pd.util.hash_pandas_object(s, index=True).values
    return hashlib.sha256(buf.tobytes()).hexdigest()[:16]


def _get_package_versions() -> dict[str, str]:
    """Snapshot versions of key scientific packages."""
    packages = [
        "numpy", "pandas", "scipy", "sklearn", "statsmodels",
        "torch", "transformers", "matplotlib", "networkx",
    ]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return versions


def _get_seeds() -> tuple[int | None, int | None, str | None]:
    """Capture current random seeds where possible."""
    np_seed: int | None = None
    try:
        state = np.random.get_state()
        np_seed = int(state[1][0]) if len(state) > 1 else None
    except Exception:
        pass

    torch_seed: int | None = None
    try:
        import torch
        torch_seed = int(torch.initial_seed())
    except Exception:
        pass

    import os
    py_hash_seed = os.environ.get("PYTHONHASHSEED")

    return np_seed, torch_seed, py_hash_seed


def capture_bundle(
    experiment_id: str,
    session_id: str,
    cycle: int,
    expression: pd.DataFrame,
    groups: pd.Series,
    config: dict[str, Any] | None = None,
    tool_sequence: list[str] | None = None,
) -> ReproducibilityBundle:
    """Capture a full reproducibility snapshot at experiment time."""
    np_seed, torch_seed, py_seed = _get_seeds()

    return ReproducibilityBundle(
        experiment_id=experiment_id,
        session_id=session_id,
        cycle=cycle,
        expression_hash=_hash_dataframe(expression),
        groups_hash=_hash_series(groups),
        n_samples=expression.shape[1],
        n_genes=expression.shape[0],
        sample_ids=list(expression.columns.astype(str))[:50],
        numpy_seed=np_seed,
        torch_seed=torch_seed,
        python_hash_seed=py_seed,
        python_version=sys.version,
        platform_info=platform.platform(),
        package_versions=_get_package_versions(),
        config_snapshot=config or {},
        tool_sequence=tool_sequence or [],
    )


class ReproducibilityLog:
    """Persist and query reproducibility bundles."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._bundles: list[ReproducibilityBundle] = []
        if self.path.exists() and self.path.stat().st_size > 0:
            try:
                raw = json.loads(self.path.read_text())
                self._bundles = [ReproducibilityBundle(**b) for b in raw.get("bundles", [])]
            except (json.JSONDecodeError, TypeError):
                pass

    def add(self, bundle: ReproducibilityBundle) -> None:
        self._bundles.append(bundle)
        self._flush()

    def _flush(self) -> None:
        payload = {"bundles": [asdict(b) for b in self._bundles]}
        self.path.write_text(json.dumps(payload, indent=2))

    def get(self, experiment_id: str) -> ReproducibilityBundle | None:
        for b in self._bundles:
            if b.experiment_id == experiment_id:
                return b
        return None

    def for_session(self, session_id: str) -> list[ReproducibilityBundle]:
        return [b for b in self._bundles if b.session_id == session_id]

    def verify_data_match(self, experiment_id: str, expression: pd.DataFrame, groups: pd.Series) -> dict[str, bool]:
        """Check if current data matches the original experiment data."""
        bundle = self.get(experiment_id)
        if bundle is None:
            return {"found": False}
        return {
            "found": True,
            "expression_match": _hash_dataframe(expression) == bundle.expression_hash,
            "groups_match": _hash_series(groups) == bundle.groups_hash,
            "shape_match": (
                expression.shape[1] == bundle.n_samples
                and expression.shape[0] == bundle.n_genes
            ),
        }

    def all_bundles(self) -> list[ReproducibilityBundle]:
        return list(self._bundles)
