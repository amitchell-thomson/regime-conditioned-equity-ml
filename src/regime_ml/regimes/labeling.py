"""Dynamic regime labeling via archetype pool matching.

Algorithm:
  1. Load archetype pool from configs/regimes/regime_archetypes.yaml.
  2. Compute per-state weighted group Z-scores from the feature matrix.
  3. Build a cosine-similarity score matrix (n_states x n_archetypes).
  4. Solve the linear assignment problem so no archetype is used twice.
  5. Apply confidence + margin thresholds — states that fall short are
     labelled "Unclassified Regime {k}" with the top-2 candidates noted.

ANALYSIS ONLY: pass smooth_proba() output for cleanest group separation.
Never use labels derived here in trading signals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from regime_ml.data.macro import build_featuregroup_map

logger = logging.getLogger(__name__)

_GROUPS = ("rates", "inflation", "growth", "liquidity", "employment")
_DEFAULT_ARCHETYPES_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "configs/regimes/regime_archetypes.yaml"
)


def _load_archetypes(path: Path | None = None) -> tuple[dict, float, float]:
    """Load archetype signatures and labelling thresholds from YAML.

    Returns:
        archetypes: {key: {"display_name": str, "signatures": {group: float}}}
        min_confidence: minimum cosine similarity for a named label
        min_margin: minimum (best - runner_up) score gap to confirm assignment
    """
    cfg_path = Path(path) if path is not None else _DEFAULT_ARCHETYPES_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"regime_archetypes.yaml not found at {cfg_path}. "
            "Create it or pass an explicit archetypes_path."
        )
    with open(cfg_path, "r") as fh:
        raw = yaml.safe_load(fh)

    label_cfg = raw.get("label_config", {})
    min_confidence = float(label_cfg.get("min_confidence", 0.45))
    min_margin = float(label_cfg.get("min_margin", 0.08))
    archetypes = raw.get("archetypes", {})

    if not archetypes:
        raise ValueError("regime_archetypes.yaml contains no archetypes.")

    return archetypes, min_confidence, min_margin


def _build_signature_matrix(
    archetypes: dict, groups: tuple[str, ...]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build the (n_archetypes, n_groups) signature matrix.

    Returns:
        A:               (n_archetypes, n_groups) array of signature values
        archetype_keys:  list of archetype keys in row order
        display_names:   list of display_names in the same order
    """
    archetype_keys = list(archetypes.keys())
    display_names = [archetypes[k]["display_name"] for k in archetype_keys]
    A = np.array(
        [
            [float(archetypes[k]["signatures"].get(g, 0.0)) for g in groups]
            for k in archetype_keys
        ],
        dtype=float,
    )
    return A, archetype_keys, display_names


def _cosine_similarity_matrix(state_vecs: np.ndarray, archetype_vecs: np.ndarray) -> np.ndarray:
    """Compute (n_states, n_archetypes) cosine similarity matrix.

    Both inputs are L2-normalised before computing the dot product so the
    result is bounded in [-1, 1].
    """
    eps = 1e-12
    sv = state_vecs / (np.linalg.norm(state_vecs, axis=1, keepdims=True) + eps)
    av = archetype_vecs / (np.linalg.norm(archetype_vecs, axis=1, keepdims=True) + eps)
    return sv @ av.T  # (n_states, n_archetypes)


def label_regimes(
    X: np.ndarray,
    proba: np.ndarray,
    feature_names: list[str],
    archetypes_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Label HMM states by matching them to economic regime archetypes.

    Args:
        X:               (T, d) feature matrix (standardised).
        proba:           (T, K) regime probability matrix. Pass smooth_proba()
                         for best group separation — analysis only.
        feature_names:   Length-d list of feature column names.
        archetypes_path: Optional override for configs/regimes/regime_archetypes.yaml.

    Returns:
        List of K dicts, one per state, with keys:
          state_idx       int   — HMM state index (0-based)
          label           str   — display name or "Unclassified Regime {k}"
          status          str   — "matched" | "unclassified"
          confidence      float — cosine similarity of best archetype match
          margin          float — confidence minus runner_up score
          archetype_key   str | None
          runner_up       str   — display name of second-best archetype
          runner_up_score float

    Notes:
        ANALYSIS ONLY. Never use the returned labels to generate trading signals.
        For live regime assignment use model.filter_proba() (causal).
    """
    archetypes, min_confidence, min_margin = _load_archetypes(archetypes_path)
    A, archetype_keys, display_names = _build_signature_matrix(archetypes, _GROUPS)

    featuregroup_map = build_featuregroup_map(feature_names)

    # Fallback for PCA-style columns (e.g. "growth_pc1", "rates_pc2") whose
    # prefix is a group name rather than a FRED series code, so they map to
    # "unknown" via build_featuregroup_map.
    _groups_set = set(_GROUPS)
    for feat in feature_names:
        if featuregroup_map.get(feat, "unknown") == "unknown":
            prefix = feat.split("_")[0]
            if prefix in _groups_set:
                featuregroup_map[feat] = prefix

    X = np.asarray(X, float)
    gamma = np.asarray(proba, float)
    T, d = X.shape
    T2, K = gamma.shape
    if T != T2:
        raise ValueError(f"X ({T} rows) and proba ({T2} rows) must align on time dimension.")
    if len(feature_names) != d:
        raise ValueError(f"feature_names length {len(feature_names)} must match X columns {d}.")

    # --- Weighted state means: mu_k (K, d)
    Nk = np.maximum(gamma.sum(axis=0), 1e-12)  # (K,)
    mu_k = (gamma.T @ X) / Nk[:, None]         # (K, d)

    # --- Group-level scores per state: state_vecs (K, n_groups)
    group_to_idx: dict[str, list[int]] = {g: [] for g in _GROUPS}
    for j, fname in enumerate(feature_names):
        g = featuregroup_map.get(fname, "unknown")
        if g in group_to_idx:
            group_to_idx[g].append(j)

    state_vecs = np.zeros((K, len(_GROUPS)), dtype=float)
    for gi, g in enumerate(_GROUPS):
        idxs = group_to_idx[g]
        if idxs:
            state_vecs[:, gi] = mu_k[:, idxs].mean(axis=1)

    # --- Cosine similarity: S[k, j] = similarity of state k to archetype j
    S = _cosine_similarity_matrix(state_vecs, A)  # (K, n_archetypes)

    # --- Linear assignment: maximise total similarity (no archetype used twice)
    n_archetypes = len(archetype_keys)
    if n_archetypes >= K:
        row_ind, col_ind = linear_sum_assignment(-S)
        assigned_archetype_idx = {int(row_ind[i]): int(col_ind[i]) for i in range(len(row_ind))}
    else:
        # More states than archetypes: assign greedily without repeat
        # (rare — archetype pool should always be > K)
        logger.warning(
            "label_regimes: more HMM states (%d) than archetypes (%d). "
            "Some states will be unclassified by design.",
            K, n_archetypes,
        )
        row_ind, col_ind = linear_sum_assignment(-S[:, :n_archetypes])
        assigned_archetype_idx = {int(row_ind[i]): int(col_ind[i]) for i in range(len(row_ind))}

    # --- Build per-state label dicts
    results: list[dict[str, Any]] = []
    for k in range(K):
        assigned_j = assigned_archetype_idx.get(k)
        best_score = float(S[k, assigned_j]) if assigned_j is not None else -np.inf

        # Runner-up: best score among un-assigned archetypes
        runner_up_j = int(np.argmax([
            S[k, j] if j != assigned_j else -np.inf
            for j in range(n_archetypes)
        ]))
        runner_up_score = float(S[k, runner_up_j])
        margin = best_score - runner_up_score

        if assigned_j is not None and best_score >= min_confidence:
            label = display_names[assigned_j]
            status = "matched"
            archetype_key: str | None = archetype_keys[assigned_j]
        else:
            label = f"Unclassified Regime {k}"
            status = "unclassified"
            archetype_key = None
            logger.info(
                "label_regimes: state %d unclassified — best score=%.3f (threshold=%.2f), "
                "best archetype=%s, runner_up=%s (score=%.3f)",
                k, best_score, min_confidence,
                display_names[assigned_j] if assigned_j is not None else "none",
                display_names[runner_up_j], runner_up_score,
            )

        results.append({
            "state_idx": k,
            "label": label,
            "status": status,
            "confidence": round(best_score, 4),
            "margin": round(margin, 4),
            "archetype_key": archetype_key,
            "runner_up": display_names[runner_up_j],
            "runner_up_score": round(runner_up_score, 4),
        })

    # Log summary
    matched = sum(1 for r in results if r["status"] == "matched")
    logger.info(
        "label_regimes: %d/%d states matched to named archetypes.", matched, K
    )
    return results
