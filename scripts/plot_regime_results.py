"""Generate regime analysis plots from the most recent pipeline run.

Loads regime assignments, features, and model metadata from data/regimes/ and
produces a set of interactive HTML plots saved to data/regimes/plots/.

Usage:
    uv run python scripts/plot_regime_results.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

from regime_ml.regimes.visualisation import (
    plot_regime_confidence,
    plot_regime_periods,
    plot_regime_timeseries,
    plot_transition_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "regimes"
FEAT_PATH = ROOT / "data" / "features" / "macro_features_ready.parquet"
EPISODES_PATH = ROOT / "configs" / "regimes" / "economic_episodes.yaml"
OUT_DIR = DATA_DIR / "plots"


def _build_regime_names(label_results: list[dict]) -> dict[int, str]:
    return {r["state_idx"]: r["label"] for r in label_results}


def _load_run_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict, list[dict]]:
    """Load regime assignments, features, transmat, metadata, and label_results."""
    assignments = pd.read_parquet(DATA_DIR / "regime_assignments.parquet")
    features = pd.read_parquet(FEAT_PATH)
    features = features.reindex(assignments.index)
    transmat = np.load(DATA_DIR / "best_model" / "transmat.npy")
    with open(DATA_DIR / "run_metadata.json") as fh:
        metadata = json.load(fh)
    label_results: list[dict] = metadata["label_results"]
    return assignments, features, transmat, metadata, label_results


def _load_episodes() -> list[dict]:
    with open(EPISODES_PATH) as fh:
        return yaml.safe_load(fh)["episodes"]


def plot_1_timeseries(
    assignments: pd.DataFrame,
    features: pd.DataFrame,
    regime_names: dict[int, str],
    metadata: dict,
) -> go.Figure:
    """3-panel: regime labels / PC features / regime probabilities."""
    proba_cols = [c for c in assignments.columns if c.startswith("filter_proba_")]
    proba = assignments[proba_cols].to_numpy()
    regimes = assignments["regime"].to_numpy()

    fig = plot_regime_timeseries(
        features=features,
        regimes=regimes,
        proba=proba,
        regime_names=regime_names,
        title="Regime Classification Over Time (filter_proba)",
    )
    train_end = str(pd.Timestamp(metadata.get("train_end_date", "2019-01-01")).date())
    for row in [1, 2, 3]:
        fig.add_shape(
            type="line", xref="x", yref="paper",
            x0=train_end, x1=train_end, y0=0, y1=1,
            line=dict(dash="dot", color="black", width=1),
            row=row, col=1,
        )
    fig.add_annotation(x=train_end, y=1.02, yref="paper", text="IS/OOS", showarrow=False, font=dict(size=10))
    return fig


def plot_2_periods(
    assignments: pd.DataFrame,
    features: pd.DataFrame,
    regime_names: dict[int, str],
) -> go.Figure:
    """Gantt-style timeline of regime periods (>= 20 days)."""
    return plot_regime_periods(
        features=features,
        regimes=assignments["regime"].to_numpy(),
        regime_names=regime_names,
        min_duration_days=20,
        title="Regime Periods (>= 20 days)",
    )


def plot_3_transition_matrix(
    transmat: np.ndarray,
    regime_names: dict[int, str],
) -> go.Figure:
    return plot_transition_matrix(
        transition_matrix=transmat,
        regime_names=regime_names,
        title="Regime Transition Probabilities (fitted HMM)",
    )


def plot_4_confidence(
    assignments: pd.DataFrame,
    features: pd.DataFrame,
    regime_names: dict[int, str],
) -> go.Figure:
    """Max filter_proba (assignment confidence) over time."""
    proba_cols = [c for c in assignments.columns if c.startswith("filter_proba_")]
    proba = assignments[proba_cols].to_numpy()
    return plot_regime_confidence(
        features=features,
        regimes=assignments["regime"].to_numpy(),
        proba=proba,
        regime_names=regime_names,
        confidence_threshold=0.7,
        title="Regime Assignment Confidence (max filter_proba)",
    )


def plot_5_episode_overlay(
    assignments: pd.DataFrame,
    episodes: list[dict],
    regime_names: dict[int, str],
    label_results: list[dict],
) -> go.Figure:
    """Regime label timeseries with economic episode shading.

    Shaded bands show canonical episode periods.  Colour indicates match status:
      green  = episode's expected archetype >= 40% of episode days
      red    = below threshold
      gray   = archetype not in this model's pool
    """
    state_to_archetype: dict[int, str | None] = {
        r["state_idx"]: r.get("archetype_key") for r in label_results
    }
    pool_name: str = label_results[0].get("pool", "canonical") if label_results else "canonical"

    canonical_to_pool: dict[str, str] = {}
    if pool_name == "n4":
        canonical_to_pool = {
            "expansion": "risk_on", "recovery": "risk_on",
            "slowdown": "contraction", "recession": "contraction",
            "stagflation": "inflation_policy", "policy_constrained": "inflation_policy",
            "liquidity_crisis": "crisis",
        }
    elif pool_name == "n3":
        canonical_to_pool = {
            "expansion": "risk_on", "recovery": "risk_on",
            "slowdown": "contraction", "recession": "contraction",
            "stagflation": "contraction", "policy_constrained": "contraction",
            "liquidity_crisis": "crisis",
        }

    regime_series = assignments["regime"]
    archetype_to_state: dict[str, int] = {
        v: k for k, v in state_to_archetype.items() if v is not None
    }

    def _episode_match(ep: dict) -> tuple[float | None, str]:
        start = pd.Timestamp(ep["start"])
        end = pd.Timestamp(ep["end"])
        mask = (regime_series.index >= start) & (regime_series.index <= end)
        if not mask.any():
            return None, "no_data"
        canon = ep["expected_archetype"]
        pool_key = canonical_to_pool.get(canon, canon) if canonical_to_pool else canon
        if pool_key not in archetype_to_state:
            return None, "not_in_model"
        expected_state = archetype_to_state[pool_key]
        pct = float((regime_series[mask] == expected_state).sum() / mask.sum())
        return pct, "matched" if pct >= 0.40 else "failed"

    status_color = {
        "matched": "rgba(0,180,0,0.12)",
        "failed": "rgba(220,0,0,0.12)",
        "not_in_model": "rgba(150,150,150,0.10)",
        "no_data": "rgba(200,200,200,0.08)",
    }

    label_series = assignments["regime_label"]
    label_to_int: dict[str, int] = {v: k for k, v in regime_names.items()}

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=assignments.index,
            y=[label_to_int.get(lbl, -1) for lbl in label_series],
            mode="lines",
            name="Regime",
            line=dict(width=2, shape="hv", color="steelblue"),
            hovertemplate="%{text}<br>%{x}<extra></extra>",
            text=label_series.tolist(),
        )
    )

    for ep in episodes:
        pct, status = _episode_match(ep)
        pct_str = f"{pct:.0%}" if pct is not None else status
        label_str = f"{ep['name']} ({ep['expected_archetype']}, {pct_str})"
        fig.add_vrect(
            x0=ep["start"],
            x1=ep["end"],
            fillcolor=status_color[status],
            layer="below",
            line_width=0,
            annotation_text=ep["name"],
            annotation_position="top left",
            annotation=dict(font_size=8, textangle=-90),
        )
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=8, color=status_color[status].replace("0.12", "0.6").replace("0.10", "0.6").replace("0.08", "0.6")),
                name=label_str,
                showlegend=True,
            )
        )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(regime_names.keys()),
        ticktext=list(regime_names.values()),
    )
    fig.update_layout(
        title=f"Regime Assignments with Economic Episode Overlay (pool={pool_name})",
        xaxis_title="Date",
        yaxis_title="Regime",
        height=600,
        hovermode="x unified",
        legend=dict(orientation="v", x=1.02, y=1, font=dict(size=9)),
    )
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    assignments, features, transmat, metadata, label_results = _load_run_data()
    episodes = _load_episodes()
    regime_names = _build_regime_names(label_results)

    model_id = metadata.get("best_model_id", "?")
    n_regimes = metadata.get("n_regimes", "?")
    score = metadata.get("final_score", 0.0)
    logger.info("Model: %s | n_regimes=%s | score=%.4f", model_id, n_regimes, score)
    logger.info("Regime names: %s", regime_names)

    plots: list[tuple[str, go.Figure]] = [
        ("regime_timeseries.html", plot_1_timeseries(assignments, features, regime_names, metadata)),
        ("regime_periods.html", plot_2_periods(assignments, features, regime_names)),
        ("transition_matrix.html", plot_3_transition_matrix(transmat, regime_names)),
        ("regime_confidence.html", plot_4_confidence(assignments, features, regime_names)),
        ("episode_overlay.html", plot_5_episode_overlay(assignments, episodes, regime_names, label_results)),
    ]

    for filename, fig in plots:
        out_path = OUT_DIR / filename
        fig.write_html(str(out_path))
        logger.info("Saved: %s", out_path)

    logger.info("Done. Open files in data/regimes/plots/ to view.")


if __name__ == "__main__":
    main()
