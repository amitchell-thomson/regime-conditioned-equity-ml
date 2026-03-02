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
            "match_status",
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
    def test_archetype_match_true_when_expected_pct_above_threshold(self, tmp_path):
        """100% of days in expected state → expected_pct=1.0 ≥ 0.40 → match=True."""
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

        idx = pd.bdate_range("2005-01-01", periods=1000)
        regimes = pd.Series(0, index=idx, name="regime")
        lr = _label_results({0: "expansion", 1: "recession"})

        df = validate_against_episodes(regimes, lr, episodes_path=ep_path)
        assert df.loc[0, "archetype_match"] == True
        assert df.loc[0, "match_status"] == "matched"

    def test_archetype_match_false_when_expected_pct_zero(self, tmp_path):
        """0% of days in expected state → expected_pct=0.0 < 0.40 → match=False."""
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
        regimes = pd.Series(1, index=idx, name="regime")  # all in recession, not expansion
        lr = _label_results({0: "expansion", 1: "recession"})

        df = validate_against_episodes(regimes, lr, episodes_path=ep_path)
        assert df.loc[0, "archetype_match"] == False
        assert df.loc[0, "match_status"] == "failed"

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
        assert df.loc[0, "match_status"] == "not_in_model"
        assert df.loc[0, "archetype_match"] == False

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


# ---------------------------------------------------------------------------
# Threshold-based match logic
# ---------------------------------------------------------------------------


def _write_archetypes_with_threshold(tmp_path: Path, threshold: float) -> Path:
    """Write a minimal archetypes YAML with a custom episode_match_threshold."""
    content = textwrap.dedent(
        f"""\
        label_config:
          min_confidence: 0.0
          min_margin: 0.0
          episode_match_threshold: {threshold}
        groups: [growth, inflation]
        archetypes:
          expansion:
            display_name: "Expansion"
            signatures: {{growth: 1.0, inflation: -0.5}}
          recession:
            display_name: "Recession"
            signatures: {{growth: -1.0, inflation: 0.5}}
    """
    )
    p = tmp_path / "arch_threshold.yaml"
    p.write_text(content)
    return p


class TestThresholdBasedMatch:
    def test_match_exactly_at_threshold(self, tmp_path):
        """expected_pct exactly equal to threshold should be a match."""
        arch_path = _write_archetypes_with_threshold(tmp_path, threshold=0.40)
        yaml_text = textwrap.dedent(
            """\
            episodes:
              - name: "Borderline"
                start: "2005-01-03"
                end: "2005-12-30"
                expected_archetype: "expansion"
        """
        )
        ep_path = _write_episodes_yaml(tmp_path, yaml_text)

        # Build exactly 40% expansion, 60% recession days.
        idx = pd.bdate_range("2005-01-01", periods=100)
        vals = np.array([0] * 40 + [1] * 60)
        regimes = pd.Series(vals, index=idx, name="regime")
        lr = _label_results({0: "expansion", 1: "recession"})

        df = validate_against_episodes(regimes, lr, episodes_path=ep_path, archetypes_path=arch_path)
        assert df.loc[0, "expected_pct"] == pytest.approx(0.40, abs=0.01)
        assert df.loc[0, "archetype_match"] == True
        assert df.loc[0, "match_status"] == "matched"

    def test_match_fails_just_below_threshold(self, tmp_path):
        """expected_pct just below threshold should be a failure."""
        arch_path = _write_archetypes_with_threshold(tmp_path, threshold=0.40)
        yaml_text = textwrap.dedent(
            """\
            episodes:
              - name: "Just Below"
                start: "2005-01-03"
                end: "2005-12-30"
                expected_archetype: "expansion"
        """
        )
        ep_path = _write_episodes_yaml(tmp_path, yaml_text)

        idx = pd.bdate_range("2005-01-01", periods=100)
        vals = np.array([0] * 30 + [1] * 70)  # 30% expansion < 40% threshold
        regimes = pd.Series(vals, index=idx, name="regime")
        lr = _label_results({0: "expansion", 1: "recession"})

        df = validate_against_episodes(regimes, lr, episodes_path=ep_path, archetypes_path=arch_path)
        assert df.loc[0, "expected_pct"] == pytest.approx(0.30, abs=0.01)
        assert df.loc[0, "archetype_match"] == False
        assert df.loc[0, "match_status"] == "failed"

    def test_match_uses_configurable_threshold(self, tmp_path):
        """A high threshold (0.80) should produce fewer matches than a low one (0.20)."""
        yaml_episodes = textwrap.dedent(
            """\
            episodes:
              - name: "Mixed"
                start: "2005-01-03"
                end: "2005-12-30"
                expected_archetype: "expansion"
        """
        )
        ep_path = _write_episodes_yaml(tmp_path, yaml_episodes)

        idx = pd.bdate_range("2005-01-01", periods=100)
        vals = np.array([0] * 50 + [1] * 50)  # 50% expansion
        regimes = pd.Series(vals, index=idx, name="regime")
        lr = _label_results({0: "expansion", 1: "recession"})

        # Write two different arch files with different thresholds (use unique names).
        high_path = tmp_path / "arch_threshold_high.yaml"
        low_path = tmp_path / "arch_threshold_low.yaml"
        for fpath, thr in [(high_path, 0.80), (low_path, 0.20)]:
            content = textwrap.dedent(
                f"""\
                label_config:
                  min_confidence: 0.0
                  min_margin: 0.0
                  episode_match_threshold: {thr}
                groups: [growth, inflation]
                archetypes:
                  expansion:
                    display_name: "Expansion"
                    signatures: {{growth: 1.0, inflation: -0.5}}
                  recession:
                    display_name: "Recession"
                    signatures: {{growth: -1.0, inflation: 0.5}}
            """
            )
            fpath.write_text(content)

        df_high = validate_against_episodes(regimes, lr, episodes_path=ep_path, archetypes_path=high_path)
        assert df_high.loc[0, "archetype_match"] == False  # 50% < 80%

        df_low = validate_against_episodes(regimes, lr, episodes_path=ep_path, archetypes_path=low_path)
        assert df_low.loc[0, "archetype_match"] == True  # 50% >= 20%


# ---------------------------------------------------------------------------
# not_in_model status
# ---------------------------------------------------------------------------


class TestNotInModel:
    def test_not_in_model_when_archetype_absent(self, tmp_path):
        """Episode expecting an archetype not assigned to any state → not_in_model."""
        yaml_text = textwrap.dedent(
            """\
            episodes:
              - name: "Recession Episode"
                start: "2005-01-01"
                end: "2005-12-31"
                expected_archetype: "recession"
        """
        )
        ep_path = _write_episodes_yaml(tmp_path, yaml_text)
        regimes = _make_regimes_series()
        # No state labeled "recession" — simulates 5-state model with no recession archetype.
        lr = _label_results({0: "expansion", 1: "slowdown", 2: "liquidity_crisis"})

        df = validate_against_episodes(regimes, lr, episodes_path=ep_path)
        assert df.loc[0, "match_status"] == "not_in_model"
        assert df.loc[0, "archetype_match"] == False
        assert np.isnan(df.loc[0, "expected_pct"])

    def test_not_in_model_does_not_count_as_failure(self, tmp_path):
        """Logging summary should exclude not_in_model from applicable count."""
        yaml_text = textwrap.dedent(
            """\
            episodes:
              - name: "Matched"
                start: "2005-01-03"
                end: "2005-06-30"
                expected_archetype: "expansion"
              - name: "Not In Model"
                start: "2007-01-01"
                end: "2007-12-31"
                expected_archetype: "recession"
        """
        )
        ep_path = _write_episodes_yaml(tmp_path, yaml_text)

        idx = pd.bdate_range("2005-01-01", periods=3000)
        regimes = pd.Series(0, index=idx, name="regime")
        lr = _label_results({0: "expansion", 1: "slowdown"})

        df = validate_against_episodes(regimes, lr, episodes_path=ep_path)
        assert df.loc[0, "match_status"] == "matched"
        assert df.loc[1, "match_status"] == "not_in_model"
        # archetype_match counts: 1 matched, 1 not_in_model → archetype_match sum = 1
        assert int(df["archetype_match"].sum()) == 1


# ---------------------------------------------------------------------------
# Canonical-to-coarse mapping
# ---------------------------------------------------------------------------


def _write_archetypes_with_n3_pool(tmp_path: Path) -> Path:
    """Write archetypes YAML with taxonomy canonical_to_coarse and n3 pool."""
    content = textwrap.dedent(
        """\
        label_config:
          min_confidence: 0.0
          min_margin: 0.0
          episode_match_threshold: 0.40
        groups: [growth, inflation, rates]
        archetypes:
          expansion:
            display_name: "Expansion"
            signatures: {growth: 1.2, inflation: -0.3, rates: 0.4}
          recovery:
            display_name: "Recovery"
            signatures: {growth: 0.7, inflation: -0.2, rates: 0.7}
          recession:
            display_name: "Recession"
            signatures: {growth: -1.2, inflation: 0.1, rates: -0.4}
          liquidity_crisis:
            display_name: "Liquidity Crisis"
            signatures: {growth: -1.0, inflation: -0.3, rates: 1.5}
        taxonomy:
          canonical_to_coarse:
            n3:
              expansion: risk_on
              recovery: risk_on
              recession: contraction
              liquidity_crisis: crisis
        pools:
          n3:
            risk_on:
              display_name: "Risk-On"
              canonical_members: [expansion, recovery]
              signatures: {growth: 1.0, inflation: -0.2, rates: 0.5}
            contraction:
              display_name: "Contraction"
              canonical_members: [recession]
              signatures: {growth: -1.2, inflation: 0.1, rates: -0.4}
            crisis:
              display_name: "Liquidity Crisis"
              canonical_members: [liquidity_crisis]
              signatures: {growth: -1.0, inflation: -0.3, rates: 1.5}
    """
    )
    p = tmp_path / "archetypes_n3.yaml"
    p.write_text(content)
    return p


class TestCanonicalToCoarse:
    def test_canonical_recession_maps_to_n3_contraction(self, tmp_path):
        """For n3 pool, episode expecting 'recession' should check the 'contraction' state."""
        arch_path = _write_archetypes_with_n3_pool(tmp_path)
        yaml_text = textwrap.dedent(
            """\
            episodes:
              - name: "Recession Period"
                start: "2005-01-03"
                end: "2005-12-30"
                expected_archetype: "recession"
        """
        )
        ep_path = _write_episodes_yaml(tmp_path, yaml_text)

        idx = pd.bdate_range("2005-01-01", periods=260)
        # All days in state 1 = "contraction" (the n3 pool equivalent of recession)
        regimes = pd.Series(1, index=idx, name="regime")
        # n3 pool label_results: state 1 = contraction, pool = n3
        lr = [
            {
                "state_idx": 0,
                "label": "Risk-On",
                "status": "matched",
                "confidence": 0.9,
                "margin": 0.3,
                "margin_warning": False,
                "archetype_key": "risk_on",
                "runner_up": "Contraction",
                "runner_up_score": 0.6,
                "pool": "n3",
            },
            {
                "state_idx": 1,
                "label": "Contraction",
                "status": "matched",
                "confidence": 0.85,
                "margin": 0.25,
                "margin_warning": False,
                "archetype_key": "contraction",
                "runner_up": "Risk-On",
                "runner_up_score": 0.6,
                "pool": "n3",
            },
        ]
        df = validate_against_episodes(regimes, lr, episodes_path=ep_path, archetypes_path=arch_path)
        # "recession" → canonical_to_coarse → "contraction" → state 1 → 100% match
        assert df.loc[0, "archetype_match"] == True
        assert df.loc[0, "match_status"] == "matched"

    def test_canonical_pool_no_mapping_applied(self, tmp_path):
        """For canonical pool (no 'pool' field in label_results), no coarse mapping is applied."""
        arch_path = _write_archetypes_with_n3_pool(tmp_path)
        yaml_text = textwrap.dedent(
            """\
            episodes:
              - name: "Recession"
                start: "2005-01-03"
                end: "2005-12-30"
                expected_archetype: "recession"
        """
        )
        ep_path = _write_episodes_yaml(tmp_path, yaml_text)

        idx = pd.bdate_range("2005-01-01", periods=260)
        regimes = pd.Series(1, index=idx, name="regime")
        # Canonical pool: state 1 labeled "recession" directly (no pool field → canonical)
        lr = _label_results({0: "expansion", 1: "recession"})

        df = validate_against_episodes(regimes, lr, episodes_path=ep_path, archetypes_path=arch_path)
        # Direct lookup: "recession" → state 1 → 100% match
        assert df.loc[0, "archetype_match"] == True
        assert df.loc[0, "match_status"] == "matched"
