"""Tests for the dynamic archetype-pool labeling in regimes/labeling.py."""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from regime_ml.regimes.labeling import label_regimes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_archetypes_yaml(tmp_path: Path) -> Path:
    """Write a minimal, deterministic archetypes YAML for testing."""
    content = textwrap.dedent(
        """\
        label_config:
          min_confidence: 0.30
          min_margin: 0.05

        archetypes:
          alpha:
            display_name: "Alpha Regime"
            signatures:
              growth: 1.5
              employment: 0.5
              liquidity: 0.8
              inflation: -0.5
              rates: -0.3

          beta:
            display_name: "Beta Regime"
            signatures:
              growth: -1.5
              employment: -0.6
              liquidity: -1.0
              inflation: 0.3
              rates: 0.2

          gamma:
            display_name: "Gamma Regime"
            signatures:
              inflation: 1.8
              rates: 1.2
              growth: -0.5
              liquidity: -0.4
              employment: -0.3

          delta:
            display_name: "Delta Regime"
            signatures:
              liquidity: -1.8
              growth: -0.8
              employment: -0.5
              inflation: -0.2
              rates: -0.1

          epsilon:
            display_name: "Epsilon Regime"
            signatures:
              growth: 0.9
              employment: 0.7
              liquidity: 0.5
              inflation: -0.3
              rates: -0.5

          zeta:
            display_name: "Zeta Regime"
            signatures:
              rates: 1.5
              inflation: 1.0
              growth: 0.2
              liquidity: -0.6
              employment: -0.1
    """
    )
    p = tmp_path / "archetypes_test.yaml"
    p.write_text(content)
    return p


def _make_feature_data(T: int, K: int, d: int = 6, seed: int = 0) -> tuple:
    """Create synthetic (X, proba, feature_names) where each state has a clear
    dominant group pattern driven by the feature layout.

    Groups (with d=6, 2 features each — or 1 each for d=5):
      growth: f0, f1 (if d>=2)
      employment: f2
      liquidity: f3
      inflation: f4
      rates: f5 (if d>=6)
    """
    rng = np.random.default_rng(seed)
    # One-hot proba — state k dominates block of rows
    proba = np.zeros((T, K))
    block = T // K
    for k in range(K):
        start = k * block
        end = (k + 1) * block if k < K - 1 else T
        proba[start:end, k] = 1.0

    X = rng.standard_normal((T, d)) * 0.05  # small noise baseline
    feature_names = [f"f{i}" for i in range(d)]
    return X, proba, feature_names


def _make_pca_features(T: int, K: int, seed: int = 0) -> tuple:
    """Create (X, proba, feature_names) using PCA-style group column names
    so build_featuregroup_map() returns meaningful groups.

    Column names: growth_pc1, employment_pc1, liquidity_pc1, inflation_pc1, rates_pc1, rates_pc2
    """
    feature_names = [
        "growth_pc1",
        "employment_pc1",
        "liquidity_pc1",
        "inflation_pc1",
        "rates_pc1",
        "rates_pc2",
    ]
    d = len(feature_names)
    rng = np.random.default_rng(seed)
    T_per_state = T // K

    # Give each state a distinct group profile
    state_means = {
        0: {
            "growth_pc1": 1.5,
            "employment_pc1": 0.8,
            "liquidity_pc1": 1.0,
            "inflation_pc1": -0.5,
            "rates_pc1": -0.3,
            "rates_pc2": -0.2,
        },
        1: {
            "growth_pc1": -1.5,
            "employment_pc1": -0.8,
            "liquidity_pc1": -1.2,
            "inflation_pc1": 0.3,
            "rates_pc1": -0.2,
            "rates_pc2": 0.1,
        },
        2: {
            "growth_pc1": -0.5,
            "employment_pc1": -0.3,
            "liquidity_pc1": -0.4,
            "inflation_pc1": 1.8,
            "rates_pc1": 1.3,
            "rates_pc2": 0.9,
        },
    }
    X_blocks = []
    proba_blocks = []
    for k in range(K):
        n_k = T_per_state if k < K - 1 else T - (K - 1) * T_per_state
        means = [state_means.get(k, {}).get(fn, 0.0) for fn in feature_names]
        block = rng.standard_normal((n_k, d)) * 0.1 + np.array(means)
        X_blocks.append(block)
        p_block = np.zeros((n_k, K))
        p_block[:, k] = 1.0
        proba_blocks.append(p_block)

    X = np.vstack(X_blocks)
    proba = np.vstack(proba_blocks)
    return X, proba, feature_names


# ---------------------------------------------------------------------------
# Core output structure tests
# ---------------------------------------------------------------------------


class TestLabelRegimesOutputStructure:
    def test_returns_list_of_dicts(self, tmp_path):
        arch_path = _synthetic_archetypes_yaml(tmp_path)
        X, proba, names = _make_pca_features(T=120, K=3)
        results = label_regimes(X, proba, names, archetypes_path=arch_path)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_required_keys_present(self, tmp_path):
        arch_path = _synthetic_archetypes_yaml(tmp_path)
        X, proba, names = _make_pca_features(T=120, K=3)
        results = label_regimes(X, proba, names, archetypes_path=arch_path)
        required = {
            "state_idx",
            "label",
            "status",
            "confidence",
            "margin",
            "margin_warning",
            "archetype_key",
            "runner_up",
            "runner_up_score",
            "pool",
        }
        for r in results:
            assert required <= set(r.keys()), f"Missing keys in {r}"

    def test_state_idx_matches_position(self, tmp_path):
        arch_path = _synthetic_archetypes_yaml(tmp_path)
        X, proba, names = _make_pca_features(T=120, K=3)
        results = label_regimes(X, proba, names, archetypes_path=arch_path)
        for i, r in enumerate(results):
            assert r["state_idx"] == i

    def test_confidence_in_range(self, tmp_path):
        arch_path = _synthetic_archetypes_yaml(tmp_path)
        X, proba, names = _make_pca_features(T=120, K=3)
        results = label_regimes(X, proba, names, archetypes_path=arch_path)
        for r in results:
            assert -1.0 <= r["confidence"] <= 1.0, f"confidence={r['confidence']} out of [-1,1]"

    def test_margin_nonnegative(self, tmp_path):
        arch_path = _synthetic_archetypes_yaml(tmp_path)
        X, proba, names = _make_pca_features(T=120, K=3)
        results = label_regimes(X, proba, names, archetypes_path=arch_path)
        for r in results:
            assert r["margin"] >= -1e-6, f"margin={r['margin']} is negative"


# ---------------------------------------------------------------------------
# No duplicate archetype assignments
# ---------------------------------------------------------------------------


class TestNoDuplicateArchetypes:
    def test_no_duplicate_archetypes_k3(self, tmp_path):
        arch_path = _synthetic_archetypes_yaml(tmp_path)
        X, proba, names = _make_pca_features(T=300, K=3)
        results = label_regimes(X, proba, names, archetypes_path=arch_path)
        assigned_keys = [r["archetype_key"] for r in results if r["archetype_key"] is not None]
        assert len(assigned_keys) == len(set(assigned_keys)), f"Duplicate archetype assignments: {assigned_keys}"

    def test_no_duplicate_archetypes_k4(self, tmp_path):
        arch_path = _synthetic_archetypes_yaml(tmp_path)
        # K=4 states, 6 archetypes available — should still have no duplicates
        T = 400
        K = 4
        rng = np.random.default_rng(7)
        block = T // K
        X = rng.standard_normal((T, 6)) * 0.1
        proba = np.zeros((T, K))
        for k in range(K):
            proba[k * block : (k + 1) * block, k] = 1.0
        names = [
            "growth_pc1",
            "employment_pc1",
            "liquidity_pc1",
            "inflation_pc1",
            "rates_pc1",
            "rates_pc2",
        ]
        results = label_regimes(X, proba, names, archetypes_path=arch_path)
        assigned_keys = [r["archetype_key"] for r in results if r["archetype_key"] is not None]
        assert len(assigned_keys) == len(set(assigned_keys))


# ---------------------------------------------------------------------------
# Confidence threshold (unclassified behaviour)
# ---------------------------------------------------------------------------


class TestConfidenceThreshold:
    def test_unclassified_label_string(self, tmp_path):
        """A state with very low confidence should produce 'Unclassified Regime k'."""
        content = textwrap.dedent(
            """\
            label_config:
              min_confidence: 0.9999   # impossibly high threshold
              min_margin: 0.05
            archetypes:
              only_one:
                display_name: "Only One"
                signatures:
                  growth: 1.0
                  employment: 0.0
                  liquidity: 0.0
                  inflation: 0.0
                  rates: 0.0
        """
        )
        p = tmp_path / "strict_archetypes.yaml"
        p.write_text(content)

        rng = np.random.default_rng(0)
        T, K, d = 100, 1, 6
        X = rng.standard_normal((T, d)) * 0.1
        proba = np.ones((T, K))
        names = [
            "growth_pc1",
            "employment_pc1",
            "liquidity_pc1",
            "inflation_pc1",
            "rates_pc1",
            "rates_pc2",
        ]
        results = label_regimes(X, proba, names, archetypes_path=p)
        assert results[0]["status"] == "unclassified"
        assert results[0]["label"].startswith("Unclassified Regime")
        assert results[0]["archetype_key"] is None

    def test_matched_label_when_above_threshold(self, tmp_path):
        arch_path = _synthetic_archetypes_yaml(tmp_path)
        X, proba, names = _make_pca_features(T=300, K=3)
        results = label_regimes(X, proba, names, archetypes_path=arch_path)
        # With well-separated synthetic states at least one should be matched
        matched = [r for r in results if r["status"] == "matched"]
        assert len(matched) >= 1, "Expected at least one matched state with well-separated data"


# ---------------------------------------------------------------------------
# Works for varying K values
# ---------------------------------------------------------------------------


class TestVaryingK:
    def test_k_equals_2(self, tmp_path):
        arch_path = _synthetic_archetypes_yaml(tmp_path)
        X, proba, names = _make_pca_features(T=200, K=2)
        results = label_regimes(X, proba, names, archetypes_path=arch_path)
        assert len(results) == 2

    def test_k_equals_5(self, tmp_path):
        arch_path = _synthetic_archetypes_yaml(tmp_path)
        T, K = 500, 5
        rng = np.random.default_rng(99)
        block = T // K
        X = rng.standard_normal((T, 6)) * 0.1
        proba = np.zeros((T, K))
        for k in range(K):
            proba[k * block : (k + 1) * block, k] = 1.0
        names = [
            "growth_pc1",
            "employment_pc1",
            "liquidity_pc1",
            "inflation_pc1",
            "rates_pc1",
            "rates_pc2",
        ]
        results = label_regimes(X, proba, names, archetypes_path=arch_path)
        assert len(results) == 5
        assigned_keys = [r["archetype_key"] for r in results if r["archetype_key"] is not None]
        assert len(assigned_keys) == len(set(assigned_keys)), "Duplicate archetype keys for K=5"

    def test_k_greater_than_archetypes_handled(self, tmp_path):
        """When K > n_archetypes, excess states should be unclassified, not crash."""
        content = textwrap.dedent(
            """\
            label_config:
              min_confidence: 0.0
              min_margin: 0.0
            archetypes:
              one:
                display_name: "One"
                signatures:
                  growth: 1.0
                  employment: 0.0
                  liquidity: 0.0
                  inflation: 0.0
                  rates: 0.0
              two:
                display_name: "Two"
                signatures:
                  growth: -1.0
                  employment: 0.0
                  liquidity: 0.0
                  inflation: 0.0
                  rates: 0.0
        """
        )
        p = tmp_path / "two_archetypes.yaml"
        p.write_text(content)

        T, K = 300, 4  # 4 states, only 2 archetypes
        rng = np.random.default_rng(5)
        X = rng.standard_normal((T, 6)) * 0.1
        proba = np.zeros((T, K))
        block = T // K
        for k in range(K):
            proba[k * block : (k + 1) * block, k] = 1.0
        names = [
            "growth_pc1",
            "employment_pc1",
            "liquidity_pc1",
            "inflation_pc1",
            "rates_pc1",
            "rates_pc2",
        ]
        results = label_regimes(X, proba, names, archetypes_path=p)
        assert len(results) == K
        # Extra states beyond 2 archetypes must be unclassified
        unclassified = [r for r in results if r["archetype_key"] is None]
        assert len(unclassified) >= K - 2


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_mismatched_T_raises(self, tmp_path):
        arch_path = _synthetic_archetypes_yaml(tmp_path)
        X = np.random.randn(100, 6)
        proba = np.random.dirichlet([1, 1, 1], size=50)
        names = [
            "growth_pc1",
            "employment_pc1",
            "liquidity_pc1",
            "inflation_pc1",
            "rates_pc1",
            "rates_pc2",
        ]
        with pytest.raises(ValueError, match="time dimension"):
            label_regimes(X, proba, names, archetypes_path=arch_path)

    def test_missing_archetypes_file_raises(self, tmp_path):
        X = np.random.randn(50, 6)
        proba = np.ones((50, 2)) / 2
        names = [
            "growth_pc1",
            "employment_pc1",
            "liquidity_pc1",
            "inflation_pc1",
            "rates_pc1",
            "rates_pc2",
        ]
        bad_path = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            label_regimes(X, proba, names, archetypes_path=bad_path)


# ---------------------------------------------------------------------------
# margin_warning field
# ---------------------------------------------------------------------------


def _archetypes_yaml_with_margin_threshold(tmp_path: Path, threshold: float) -> Path:
    """Write archetypes YAML with a configurable margin_warning_threshold."""
    content = textwrap.dedent(
        f"""\
        label_config:
          min_confidence: 0.0
          min_margin: 0.0
          margin_warning_threshold: {threshold}
        groups:
          - growth
          - employment
          - liquidity
          - inflation
          - rates
        archetypes:
          alpha:
            display_name: "Alpha"
            signatures:
              growth: 1.5
              employment: 0.5
              liquidity: 0.8
              inflation: -0.5
              rates: -0.3
          beta:
            display_name: "Beta"
            signatures:
              growth: -1.5
              employment: -0.6
              liquidity: -1.0
              inflation: 0.3
              rates: 0.2
    """
    )
    p = tmp_path / "archetypes_margin.yaml"
    p.write_text(content)
    return p


class TestMarginWarning:
    def test_margin_warning_is_bool(self, tmp_path):
        arch_path = _synthetic_archetypes_yaml(tmp_path)
        X, proba, names = _make_pca_features(T=120, K=3)
        results = label_regimes(X, proba, names, archetypes_path=arch_path)
        for r in results:
            assert isinstance(r["margin_warning"], bool), (
                f"margin_warning should be bool, got {type(r['margin_warning'])}"
            )

    def test_margin_warning_false_when_margin_exceeds_threshold(self, tmp_path):
        """State with large margin should not trigger warning."""
        arch_path = _archetypes_yaml_with_margin_threshold(tmp_path, threshold=0.01)
        X, proba, names = _make_pca_features(T=300, K=2)
        results = label_regimes(X, proba, names, archetypes_path=arch_path)
        matched = [r for r in results if r["status"] == "matched"]
        # With threshold=0.01 and well-separated states, all matched states should
        # have margin > 0.01 → no warnings.
        for r in matched:
            assert not r["margin_warning"], (
                f"Expected no margin_warning for margin={r['margin']} with threshold=0.01"
            )

    def test_margin_warning_true_for_nearly_identical_archetypes(self, tmp_path):
        """States matched to nearly identical archetypes should have a small margin → warning."""
        # Two archetypes differ only in 'rates'; cosine similarity to both will be very close.
        content = textwrap.dedent(
            """\
            label_config:
              min_confidence: 0.0
              min_margin: 0.0
              margin_warning_threshold: 0.20
            groups:
              - growth
              - employment
              - liquidity
              - inflation
              - rates
            archetypes:
              almost_a:
                display_name: "Almost A"
                signatures:
                  growth: 1.0
                  employment: 1.0
                  liquidity: 1.0
                  inflation: 1.0
                  rates: 1.0
              almost_b:
                display_name: "Almost B"
                signatures:
                  growth: 1.0
                  employment: 1.0
                  liquidity: 1.0
                  inflation: 1.0
                  rates: 0.99
        """
        )
        p = tmp_path / "archetypes_nearly_identical.yaml"
        p.write_text(content)

        rng = np.random.default_rng(42)
        T, K, d = 200, 2, 5
        X = rng.standard_normal((T, d)) * 0.01 + 1.0  # all features ~+1 → both archetypes are close
        proba = np.zeros((T, K))
        proba[:100, 0] = 1.0
        proba[100:, 1] = 1.0
        names = ["growth_pc1", "employment_pc1", "liquidity_pc1", "inflation_pc1", "rates_pc1"]
        results = label_regimes(X, proba, names, archetypes_path=p)
        matched = [r for r in results if r["status"] == "matched"]
        assert len(matched) >= 1, "Expected at least one matched state"
        for r in matched:
            assert r["margin_warning"], (
                f"Expected margin_warning=True for margin={r['margin']} (near-identical archetypes)"
            )


# ---------------------------------------------------------------------------
# Pool selection by K
# ---------------------------------------------------------------------------


def _archetypes_yaml_with_pools(tmp_path: Path) -> Path:
    """Write archetypes YAML with canonical + n3 + n4 pools for pool-selection tests."""
    content = textwrap.dedent(
        """\
        label_config:
          min_confidence: 0.0
          min_margin: 0.0
          margin_warning_threshold: 0.05
        groups:
          - growth
          - employment
          - liquidity
          - inflation
          - rates
        archetypes:
          arch_a:
            display_name: "Arch A"
            signatures: {growth: 1.5, employment: 0.5, liquidity: 0.8, inflation: -0.5, rates: -0.3}
          arch_b:
            display_name: "Arch B"
            signatures: {growth: -1.5, employment: -0.6, liquidity: -1.0, inflation: 0.3, rates: 0.2}
          arch_c:
            display_name: "Arch C"
            signatures: {inflation: 1.8, rates: 1.2, growth: -0.5, liquidity: -0.4, employment: -0.3}
          arch_d:
            display_name: "Arch D"
            signatures: {liquidity: -1.8, growth: -0.8, employment: -0.5, inflation: -0.2, rates: -0.1}
          arch_e:
            display_name: "Arch E"
            signatures: {growth: 0.9, employment: 0.7, liquidity: 0.5, inflation: -0.3, rates: -0.5}
        pools:
          n3:
            pool_x:
              display_name: "Pool X"
              canonical_members: [arch_a, arch_e]
              signatures: {growth: 1.2, employment: 0.6, liquidity: 0.6, inflation: -0.4, rates: -0.4}
            pool_y:
              display_name: "Pool Y"
              canonical_members: [arch_b, arch_d]
              signatures: {growth: -1.1, employment: -0.5, liquidity: -1.4, inflation: 0.0, rates: 0.0}
            pool_z:
              display_name: "Pool Z"
              canonical_members: [arch_c]
              signatures: {inflation: 1.8, rates: 1.2, growth: -0.5, liquidity: -0.4, employment: -0.3}
          n4:
            pool_p:
              display_name: "Pool P"
              canonical_members: [arch_a, arch_e]
              signatures: {growth: 1.2, employment: 0.6, liquidity: 0.6, inflation: -0.4, rates: -0.4}
            pool_q:
              display_name: "Pool Q"
              canonical_members: [arch_b]
              signatures: {growth: -1.5, employment: -0.6, liquidity: -1.0, inflation: 0.3, rates: 0.2}
            pool_r:
              display_name: "Pool R"
              canonical_members: [arch_c]
              signatures: {inflation: 1.8, rates: 1.2, growth: -0.5, liquidity: -0.4, employment: -0.3}
            pool_s:
              display_name: "Pool S"
              canonical_members: [arch_d]
              signatures: {liquidity: -1.8, growth: -0.8, employment: -0.5, inflation: -0.2, rates: -0.1}
    """
    )
    p = tmp_path / "archetypes_pools.yaml"
    p.write_text(content)
    return p


class TestPoolSelection:
    def test_k3_uses_n3_pool(self, tmp_path):
        arch_path = _archetypes_yaml_with_pools(tmp_path)
        rng = np.random.default_rng(0)
        T, K, d = 300, 3, 5
        X = rng.standard_normal((T, d))
        proba = np.zeros((T, K))
        block = T // K
        for k in range(K):
            proba[k * block : (k + 1) * block, k] = 1.0
        names = ["growth_pc1", "employment_pc1", "liquidity_pc1", "inflation_pc1", "rates_pc1"]
        results = label_regimes(X, proba, names, archetypes_path=arch_path)
        assert all(r["pool"] == "n3" for r in results), (
            f"Expected pool='n3' for K=3, got {[r['pool'] for r in results]}"
        )
        assigned_keys = {r["archetype_key"] for r in results if r["archetype_key"] is not None}
        n3_keys = {"pool_x", "pool_y", "pool_z"}
        assert assigned_keys <= n3_keys, f"n3 pool keys expected, got {assigned_keys}"

    def test_k4_uses_n4_pool(self, tmp_path):
        arch_path = _archetypes_yaml_with_pools(tmp_path)
        rng = np.random.default_rng(1)
        T, K, d = 400, 4, 5
        X = rng.standard_normal((T, d))
        proba = np.zeros((T, K))
        block = T // K
        for k in range(K):
            proba[k * block : (k + 1) * block, k] = 1.0
        names = ["growth_pc1", "employment_pc1", "liquidity_pc1", "inflation_pc1", "rates_pc1"]
        results = label_regimes(X, proba, names, archetypes_path=arch_path)
        assert all(r["pool"] == "n4" for r in results), (
            f"Expected pool='n4' for K=4, got {[r['pool'] for r in results]}"
        )
        assigned_keys = {r["archetype_key"] for r in results if r["archetype_key"] is not None}
        n4_keys = {"pool_p", "pool_q", "pool_r", "pool_s"}
        assert assigned_keys <= n4_keys, f"n4 pool keys expected, got {assigned_keys}"

    def test_k5_uses_canonical_pool(self, tmp_path):
        arch_path = _archetypes_yaml_with_pools(tmp_path)
        rng = np.random.default_rng(2)
        T, K, d = 500, 5, 5
        X = rng.standard_normal((T, d))
        proba = np.zeros((T, K))
        block = T // K
        for k in range(K):
            proba[k * block : (k + 1) * block, k] = 1.0
        names = ["growth_pc1", "employment_pc1", "liquidity_pc1", "inflation_pc1", "rates_pc1"]
        results = label_regimes(X, proba, names, archetypes_path=arch_path)
        assert all(r["pool"] == "canonical" for r in results), (
            f"Expected pool='canonical' for K=5, got {[r['pool'] for r in results]}"
        )
        canonical_keys = {"arch_a", "arch_b", "arch_c", "arch_d", "arch_e"}
        assigned_keys = {r["archetype_key"] for r in results if r["archetype_key"] is not None}
        assert assigned_keys <= canonical_keys, f"Canonical keys expected, got {assigned_keys}"

    def test_k2_uses_n3_pool(self, tmp_path):
        """K=2 should also route to n3 pool (K <= 3)."""
        arch_path = _archetypes_yaml_with_pools(tmp_path)
        rng = np.random.default_rng(3)
        T, K, d = 200, 2, 5
        X = rng.standard_normal((T, d))
        proba = np.zeros((T, K))
        proba[:100, 0] = 1.0
        proba[100:, 1] = 1.0
        names = ["growth_pc1", "employment_pc1", "liquidity_pc1", "inflation_pc1", "rates_pc1"]
        results = label_regimes(X, proba, names, archetypes_path=arch_path)
        assert all(r["pool"] == "n3" for r in results)


# ---------------------------------------------------------------------------
# Groups from YAML
# ---------------------------------------------------------------------------


class TestGroupsFromYAML:
    def test_groups_read_from_yaml(self, tmp_path):
        """Groups declared in YAML should be used; feature names must match group prefixes."""
        content = textwrap.dedent(
            """\
            label_config:
              min_confidence: 0.0
              min_margin: 0.0
              margin_warning_threshold: 0.05
            groups:
              - alpha_group
              - beta_group
            archetypes:
              pos:
                display_name: "Positive"
                signatures: {alpha_group: 1.5, beta_group: -0.5}
              neg:
                display_name: "Negative"
                signatures: {alpha_group: -1.5, beta_group: 0.5}
        """
        )
        p = tmp_path / "archetypes_groups.yaml"
        p.write_text(content)

        # Feature names use the group name as prefix → PCA fallback maps them.
        T, K, d = 200, 2, 2
        rng = np.random.default_rng(42)
        X = rng.standard_normal((T, d))
        proba = np.zeros((T, K))
        proba[:100, 0] = 1.0
        proba[100:, 1] = 1.0
        names = ["alpha_group_pc1", "beta_group_pc1"]
        # Should not raise; group mapping derived from column prefix.
        results = label_regimes(X, proba, names, archetypes_path=p)
        assert len(results) == K

    def test_groups_derived_when_absent(self, tmp_path):
        """When 'groups' key is absent from YAML, groups should be derived from signatures."""
        content = textwrap.dedent(
            """\
            label_config:
              min_confidence: 0.0
              min_margin: 0.0
            archetypes:
              alpha:
                display_name: "Alpha"
                signatures: {growth: 1.5, employment: 0.5}
              beta:
                display_name: "Beta"
                signatures: {growth: -1.5, employment: -0.5}
        """
        )
        p = tmp_path / "archetypes_no_groups.yaml"
        p.write_text(content)

        T, K, d = 200, 2, 2
        rng = np.random.default_rng(7)
        X = rng.standard_normal((T, d))
        proba = np.zeros((T, K))
        proba[:100, 0] = 1.0
        proba[100:, 1] = 1.0
        names = ["growth_pc1", "employment_pc1"]
        # Should succeed and derive groups = ["employment", "growth"] (sorted)
        results = label_regimes(X, proba, names, archetypes_path=p)
        assert len(results) == K
