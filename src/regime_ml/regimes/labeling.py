"""Dynamic regime labeling via archetype pool matching.

Algorithm:
  1. Load archetype pool from configs/regimes/regime_archetypes.yaml.
     For K <= 3 states the n3 coarsened pool is used; K == 4 uses n4; K >= 5
     uses the full canonical pool.  Pool definitions live in the YAML.
  2. Compute per-state weighted group Z-scores from the feature matrix.
  3. Build a cosine-similarity score matrix (n_states x n_archetypes).
  4. Solve the linear assignment problem so no archetype is used twice.
  5. Apply confidence threshold — states that fall short are labelled
     "Unclassified Regime {k}".  Thin margins flag margin_warning=True.

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

_DEFAULT_ARCHETYPES_PATH = Path(__file__).parent.parent.parent.parent / "configs/regimes/regime_archetypes.yaml"


def _load_archetypes(
    path: Path | None = None,
    n_states: int = 5,
) -> tuple[dict, list[str], float, float, str]:
    """Load archetype pool, groups, and thresholds from YAML.

    Pool routing:
        K <= 3  → pools.n3  (risk_on / contraction / crisis)
        K == 4  → pools.n4  (risk_on / contraction / inflation_policy / crisis)
        K >= 5  → canonical archetypes (full 7-archetype pool)

    Returns:
        archetypes:                {key: {"display_name": str, "signatures": {group: float}}}
        groups:                    ordered list of group names from YAML
        min_confidence:            minimum cosine similarity for a named label
        margin_warning_threshold:  margin below which margin_warning=True is set (informational only)
        pool_name:                 "n3" | "n4" | "canonical"
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
    margin_warning_threshold = float(label_cfg.get("margin_warning_threshold", 0.08))

    # Groups: read from YAML; fall back to deriving from canonical signature keys.
    groups: list[str] | None = raw.get("groups", None)
    if groups is None:
        all_keys: set[str] = set()
        for v in raw.get("archetypes", {}).values():
            all_keys.update(v.get("signatures", {}).keys())
        groups = sorted(all_keys)
        logger.warning(
            "label_regimes: 'groups' key not found in archetypes YAML; "
            "derived from canonical signature keys: %s",
            groups,
        )
    else:
        groups = list(groups)

    # Pool selection based on number of HMM states.
    pools = raw.get("pools", {})
    if n_states <= 3 and "n3" in pools:
        archetypes: dict = pools["n3"]
        pool_name = "n3"
    elif n_states == 4 and "n4" in pools:
        archetypes = pools["n4"]
        pool_name = "n4"
    else:
        archetypes = raw.get("archetypes", {})
        pool_name = "canonical"

    if not archetypes:
        raise ValueError(
            f"No archetypes found for n_states={n_states} (pool={pool_name!r}). "
            "Check regime_archetypes.yaml pools and archetypes sections."
        )

    return archetypes, groups, min_confidence, margin_warning_threshold, pool_name


def _build_signature_matrix(archetypes: dict, groups: tuple[str, ...]) -> tuple[np.ndarray, list[str], list[str]]:
    """Build the (n_archetypes, n_groups) signature matrix.

    Returns:
        A:               (n_archetypes, n_groups) array of signature values
        archetype_keys:  list of archetype keys in row order
        display_names:   list of display_names in the same order
    """
    archetype_keys = list(archetypes.keys())
    display_names = [archetypes[k]["display_name"] for k in archetype_keys]
    A = np.array(
        [[float(archetypes[k]["signatures"].get(g, 0.0)) for g in groups] for k in archetype_keys],
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
          state_idx           int   — HMM state index (0-based)
          label               str   — display name or "Unclassified Regime {k}"
          status              str   — "matched" | "unclassified"
          confidence          float — cosine similarity of best archetype match
          margin              float — confidence minus runner_up score
          margin_warning      bool  — True if margin < margin_warning_threshold
          archetype_key       str | None
          runner_up           str   — display name of second-best archetype
          runner_up_score     float
          pool                str   — archetype pool used: "n3" | "n4" | "canonical"

    Notes:
        ANALYSIS ONLY. Never use the returned labels to generate trading signals.
        For live regime assignment use model.filter_proba() (causal).
    """
    gamma = np.asarray(proba, float)
    T2, K = gamma.shape

    archetypes, groups, min_confidence, margin_warning_threshold, pool_name = _load_archetypes(
        archetypes_path, n_states=K
    )

    groups_tuple = tuple(groups)
    A, archetype_keys, display_names = _build_signature_matrix(archetypes, groups_tuple)

    featuregroup_map = build_featuregroup_map(feature_names)

    # Fallback for PCA-style columns (e.g. "rates_pc1", "real_economy_pc1") whose
    # prefix is a group name rather than a FRED series code.  We try each known
    # group name as a prefix so that multi-word names like "real_economy" are
    # matched correctly (simple first-token split "real_economy_pc1".split("_")[0]
    # gives "real", missing the full group name).
    groups_set = set(groups_tuple)
    for feat in feature_names:
        if featuregroup_map.get(feat, "unknown") == "unknown":
            for g in groups_set:
                if feat.startswith(f"{g}_") or feat == g:
                    featuregroup_map[feat] = g
                    break

    # Validate that every declared group has at least one feature mapped to it.
    group_to_idx_check: dict[str, list[int]] = {g: [] for g in groups_tuple}
    for j, fname in enumerate(feature_names):
        g = featuregroup_map.get(fname, "unknown")
        if g in group_to_idx_check:
            group_to_idx_check[g].append(j)
    missing_groups = [g for g, idxs in group_to_idx_check.items() if not idxs]
    if missing_groups:
        logger.warning(
            "label_regimes: groups %s have no features mapped — state vectors will be "
            "zero for these dimensions, reducing archetype match quality.",
            missing_groups,
        )

    X = np.asarray(X, float)
    T, d = X.shape
    if T != T2:
        raise ValueError(f"X ({T} rows) and proba ({T2} rows) must align on time dimension.")
    if len(feature_names) != d:
        raise ValueError(f"feature_names length {len(feature_names)} must match X columns {d}.")

    # --- Weighted state means: mu_k (K, d)
    Nk = np.maximum(gamma.sum(axis=0), 1e-12)  # (K,)
    mu_k = (gamma.T @ X) / Nk[:, None]  # (K, d)

    # --- Group-level scores per state: state_vecs (K, n_groups)
    group_to_idx: dict[str, list[int]] = {g: [] for g in groups_tuple}
    for j, fname in enumerate(feature_names):
        g = featuregroup_map.get(fname, "unknown")
        if g in group_to_idx:
            group_to_idx[g].append(j)

    state_vecs = np.zeros((K, len(groups_tuple)), dtype=float)
    for gi, g in enumerate(groups_tuple):
        idxs = group_to_idx[g]
        if idxs:
            # Use PC1 only (idxs[0]) rather than averaging all PCs.
            # Averaging PC1 and PC2 dilutes the group signal because PC2 often
            # captures a different economic dimension (e.g. rates slope vs level)
            # that partially cancels with PC1 when compared to archetype signatures.
            state_vecs[:, gi] = mu_k[:, idxs[0]]

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
            K,
            n_archetypes,
        )
        row_ind, col_ind = linear_sum_assignment(-S[:, :n_archetypes])
        assigned_archetype_idx = {int(row_ind[i]): int(col_ind[i]) for i in range(len(row_ind))}

    # --- Build per-state label dicts
    results: list[dict[str, Any]] = []
    for k in range(K):
        assigned_j = assigned_archetype_idx.get(k)
        best_score = float(S[k, assigned_j]) if assigned_j is not None else -np.inf

        # Runner-up: best score among un-assigned archetypes
        runner_up_j = int(np.argmax([S[k, j] if j != assigned_j else -np.inf for j in range(n_archetypes)]))
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
                k,
                best_score,
                min_confidence,
                display_names[assigned_j] if assigned_j is not None else "none",
                display_names[runner_up_j],
                runner_up_score,
            )

        results.append(
            {
                "state_idx": k,
                "label": label,
                "status": status,
                "confidence": round(best_score, 4),
                "margin": round(margin, 4),
                "margin_warning": bool(margin < margin_warning_threshold),
                "archetype_key": archetype_key,
                "runner_up": display_names[runner_up_j],
                "runner_up_score": round(runner_up_score, 4),
                "pool": pool_name,
            }
        )

    # Log summary
    matched = sum(1 for r in results if r["status"] == "matched")
    warnings = sum(1 for r in results if r["margin_warning"] and r["status"] == "matched")
    logger.info(
        "label_regimes: %d/%d states matched (pool=%s); %d with margin_warning.",
        matched,
        K,
        pool_name,
        warnings,
    )
    return results
