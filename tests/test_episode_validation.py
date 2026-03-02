"""Tests for validate_against_episodes() in regimes/evaluation.py."""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_ml.regimes.evaluation import validate_against_episodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_episodes_yaml(tmp_path: Path, episodes_text: str) -> Path:
    p = tmp_path / "episodes_test.yaml"
    p.write_text(episodes_text)
    return p


def _make_regimes_series(
    start: str = "2000-01-03",
    periods: int = 5000,
    pattern: list[int] | None = None,
) -> pd.Series:
    idx = pd.bdate_range(start, periods=periods)
    if pattern is None:
        # Alternate between states 0, 1, 2 roughly equally
        vals = np.tile([0, 1, 2], periods // 3 + 1)[:periods]
    else:
        vals = np.tile(pattern, periods // len(pattern) + 1)[:periods]
    return pd.Series(vals, index=idx, name="regime")


def _label_results(mapping: dict[int, str]) -> list[dict]:
    """Build a minimal label_results list from a {state_idx: archetype_key} mapping."""
    return [
        {
            "state_idx": k,
            "label": f"Label {k}",
            "status": "matched" if v is not None else "unclassified",
            "confidence": 0.8,
            "margin": 0.2,
            "archetype_key": v,
            "runner_up": "other",
            "runner_up_score": 0.3,
        }
        for k, v in mapping.items()
    ]


# ---------------------------------------------------------------------------
# Basic output structure
# ---------------------------------------------------------------------------


class TestValidateOutputStructure:
    def test_returns_dataframe(self, tmp_path):
        yaml_text = textwrap.dedent(
            """\
            episodes:
              - name: "Test Episode"
                start: "2005-01-01"
                end: "2007-12-31"
                expected_archetype: "expansion"
        """
        )
        ep_path = _write_episodes_yaml(tmp_path, yaml_text)
        regimes = _make_regimes_series()
        lr = _label_results({0: "expansion", 1: "recession", 2: None})
        result = validate_against_episodes(regimes, lr, episodes_path=ep_path)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, tmp_path):
        yaml_text = textwrap.dedent(
            """\
            episodes:
              - name: "Test"
                start: "2005-01-01"
                end: "2005-12-31"
                expected_archetype: "expansion"
        """
        )
        ep_path = _write_episodes_yaml(tmp_path, yaml_text)
        regimes = _make_regimes_series()
        lr = _label_results({0: "expansion", 1: "recession", 2: None})
        df = validate_against_episodes(regimes, lr, episodes_path=ep_path)
        expected_cols = {
            "episode",
            "start",
            "end",
            "expected_archetype",
            "dominant_state",
            "dominant_label",
            "archetype_match",
            "expected_pct",
            "dominant_pct",
            "n_days",
        }
        assert expected_cols <= set(df.columns)

    def test_one_row_per_episode(self, tmp_path):
        yaml_text = textwrap.dedent(
            """\
            episodes:
              - name: "A"
                start: "2003-01-01"
                end: "2004-12-31"
                expected_archetype: "expansion"
              - name: "B"
                start: "2007-01-01"
                end: "2008-12-31"
                expected_archetype: "recession"
              - name: "C"
                start: "2010-01-01"
                end: "2011-12-31"
                expected_archetype: "recovery"
        """
        )
        ep_path = _write_episodes_yaml(tmp_path, yaml_text)
        regimes = _make_regimes_series()
        lr = _label_results({0: "expansion", 1: "recession", 2: "recovery"})
        df = validate_against_episodes(regimes, lr, episodes_path=ep_path)
        assert len(df) == 3


# ---------------------------------------------------------------------------
# archetype_match correctness
# ---------------------------------------------------------------------------


class TestArchetypeMatch:
    def test_archetype_match_true_when_dominant_equals_expected(self, tmp_path):
        """State 0 should dominate the episode and archetype_key='expansion' → match=True."""
        yaml_text = textwrap.dedent(
            """\
            episodes:
              - name: "Expansion"
                start: "2005-01-03"
                end: "2005-12-30"
                expected_archetype: "expansion"
        """
        )
        ep_path = _write_episodes_yaml(tmp_path, yaml_text)

        # All days in episode are state 0 → dominant is state 0 → expansion → match
        idx = pd.bdate_range("2005-01-01", periods=1000)
        regimes = pd.Series(0, index=idx, name="regime")
        lr = _label_results({0: "expansion", 1: "recession"})

        df = validate_against_episodes(regimes, lr, episodes_path=ep_path)
        assert df.loc[0, "archetype_match"] is True or df.loc[0, "archetype_match"] == True

    def test_archetype_match_false_when_dominant_differs(self, tmp_path):
        """State 1 dominates but expected_archetype='expansion' (mapped to state 0) → mismatch."""
        yaml_text = textwrap.dedent(
            """\
            episodes:
              - name: "Wrong Regime"
                start: "2005-01-03"
                end: "2005-12-30"
                expected_archetype: "expansion"
        """
        )
        ep_path = _write_episodes_yaml(tmp_path, yaml_text)

        idx = pd.bdate_range("2005-01-01", periods=500)
        # Regime 1 dominates
        regimes = pd.Series(1, index=idx, name="regime")
        lr = _label_results({0: "expansion", 1: "recession"})

        df = validate_against_episodes(regimes, lr, episodes_path=ep_path)
        assert df.loc[0, "archetype_match"] is False or df.loc[0, "archetype_match"] == False

    def test_expected_pct_is_fraction_in_expected_state(self, tmp_path):
        """expected_pct should equal fraction of days in the expected archetype's state."""
        yaml_text = textwrap.dedent(
            """\
            episodes:
              - name: "Mixed"
                start: "2005-01-03"
                end: "2005-12-30"
                expected_archetype: "expansion"
        """
        )
        ep_path = _write_episodes_yaml(tmp_path, yaml_text)

        idx = pd.bdate_range("2005-01-01", periods=200)
        # 150 days state 0 (expansion), 50 days state 1
        vals = np.array([0] * 150 + [1] * 50)
        regimes = pd.Series(vals, index=idx, name="regime")
        lr = _label_results({0: "expansion", 1: "recession"})

        df = validate_against_episodes(regimes, lr, episodes_path=ep_path)
        expected_pct = df.loc[0, "expected_pct"]
        assert 0.0 <= expected_pct <= 1.0
        # State 0 = expansion; ~150 of ~200 days in episode
        assert expected_pct > 0.5, f"expected_pct={expected_pct} should be >0.5 (150 of ~200 days)"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_episode_outside_data_returns_zero_days(self, tmp_path):
        yaml_text = textwrap.dedent(
            """\
            episodes:
              - name: "Future"
                start: "2099-01-01"
                end: "2099-12-31"
                expected_archetype: "expansion"
        """
        )
        ep_path = _write_episodes_yaml(tmp_path, yaml_text)
        regimes = _make_regimes_series()
        lr = _label_results({0: "expansion"})
        df = validate_against_episodes(regimes, lr, episodes_path=ep_path)
        assert df.loc[0, "n_days"] == 0

    def test_unmatched_expected_archetype_gives_nan_expected_pct(self, tmp_path):
        """If expected_archetype not in label_results, expected_pct must be NaN."""
        yaml_text = textwrap.dedent(
            """\
            episodes:
              - name: "Test"
                start: "2005-01-01"
                end: "2005-12-31"
                expected_archetype: "unknown_archetype"
        """
        )
        ep_path = _write_episodes_yaml(tmp_path, yaml_text)
        regimes = _make_regimes_series()
        # No state is mapped to "unknown_archetype"
        lr = _label_results({0: "expansion", 1: "recession"})
        df = validate_against_episodes(regimes, lr, episodes_path=ep_path)
        assert np.isnan(df.loc[0, "expected_pct"])

    def test_missing_file_raises(self, tmp_path):
        regimes = _make_regimes_series()
        lr = _label_results({0: "expansion"})
        bad_path = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            validate_against_episodes(regimes, lr, episodes_path=bad_path)

    def test_dominant_pct_is_in_unit_interval(self, tmp_path):
        yaml_text = textwrap.dedent(
            """\
            episodes:
              - name: "Check"
                start: "2005-01-01"
                end: "2010-12-31"
                expected_archetype: "expansion"
        """
        )
        ep_path = _write_episodes_yaml(tmp_path, yaml_text)
        regimes = _make_regimes_series()
        lr = _label_results({0: "expansion", 1: "recession", 2: None})
        df = validate_against_episodes(regimes, lr, episodes_path=ep_path)
        dom = df.loc[0, "dominant_pct"]
        assert 0.0 < dom <= 1.0
