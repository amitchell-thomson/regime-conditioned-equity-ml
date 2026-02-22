"""Tests for ALFRED point-in-time reconstruction (Phase 2.4).

Fixture design:
    CFNAI series with 2 observation periods (Jan 2020, Feb 2020):

    obs_date     realtime_start  value   meaning
    2020-01-31   2020-02-28      0.10    Jan initial release
    2020-01-31   2020-03-30      0.12    Jan revision (higher)
    2020-02-29   2020-03-30      -0.05   Feb initial release
    2020-02-29   2020-04-28      -0.04   Feb revision

Point-in-time expectations:
    pub_date 2020-02-28 → value=0.10   (only Jan initial known)
    pub_date 2020-03-30 → value=-0.05  (Feb initial is more recent obs_date than Jan revision)
    pub_date 2020-04-28 → value=-0.04  (Feb revision is latest available)
"""

import pytest
import pandas as pd
import numpy as np

from regime_ml.data.macro.alignment import build_realtime_series


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CFNAI_ROWS = [
    # obs_date      realtime_start   value   series   category
    ("2020-01-31", "2020-02-28",  0.10,  "CFNAI", "CFNAI", "growth"),
    ("2020-01-31", "2020-03-30",  0.12,  "CFNAI", "CFNAI", "growth"),
    ("2020-02-29", "2020-03-30", -0.05,  "CFNAI", "CFNAI", "growth"),
    ("2020-02-29", "2020-04-28", -0.04,  "CFNAI", "CFNAI", "growth"),
]


@pytest.fixture()
def cfnai_alfred() -> pd.DataFrame:
    rows = [
        {
            "series_code": r[3],
            "date": pd.Timestamp(r[0]),
            "realtime_start": pd.Timestamp(r[1]),
            "realtime_end": pd.Timestamp("2099-12-31"),
            "value": r[2],
            "series_name": r[4],
            "category": r[5],
        }
        for r in CFNAI_ROWS
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def two_series_alfred(cfnai_alfred: pd.DataFrame) -> pd.DataFrame:
    """CFNAI + a second series (INDPRO) for filter tests."""
    indpro_rows = [
        {
            "series_code": "INDPRO",
            "date": pd.Timestamp("2020-01-31"),
            "realtime_start": pd.Timestamp("2020-02-14"),
            "realtime_end": pd.Timestamp("2099-12-31"),
            "value": 105.5,
            "series_name": "INDPRO",
            "category": "growth",
        },
        {
            "series_code": "INDPRO",
            "date": pd.Timestamp("2020-02-29"),
            "realtime_start": pd.Timestamp("2020-03-17"),
            "realtime_end": pd.Timestamp("2099-12-31"),
            "value": 104.2,
            "series_name": "INDPRO",
            "category": "growth",
        },
    ]
    return pd.concat([cfnai_alfred, pd.DataFrame(indpro_rows)], ignore_index=True)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

def test_output_schema_no_realtime_columns(cfnai_alfred):
    """Output must not contain realtime_start or realtime_end columns."""
    result = build_realtime_series(cfnai_alfred)
    assert "realtime_start" not in result.columns
    assert "realtime_end" not in result.columns


def test_output_schema_required_columns(cfnai_alfred):
    """Output must have exactly: series_code, date, value, series_name, category."""
    result = build_realtime_series(cfnai_alfred)
    required = {"series_code", "date", "value", "series_name", "category"}
    assert required.issubset(set(result.columns)), (
        f"Missing columns: {required - set(result.columns)}"
    )


# ---------------------------------------------------------------------------
# Output date = realtime_start (causal dates)
# ---------------------------------------------------------------------------

def test_output_date_column_contains_publication_dates(cfnai_alfred):
    """Output 'date' column must contain realtime_start dates, not obs_dates."""
    result = build_realtime_series(cfnai_alfred)
    cfnai = result[result["series_code"] == "CFNAI"]
    output_dates = set(cfnai["date"])
    expected_pub_dates = {
        pd.Timestamp("2020-02-28"),
        pd.Timestamp("2020-03-30"),
        pd.Timestamp("2020-04-28"),
    }
    assert output_dates == expected_pub_dates, (
        f"Expected pub dates {expected_pub_dates}, got {output_dates}"
    )


def test_output_date_not_obs_dates(cfnai_alfred):
    """obs_date (2020-01-31 and 2020-02-29) must NOT appear as output dates."""
    result = build_realtime_series(cfnai_alfred)
    assert pd.Timestamp("2020-01-31") not in set(result["date"])
    assert pd.Timestamp("2020-02-29") not in set(result["date"])


# ---------------------------------------------------------------------------
# Point-in-time correctness
# ---------------------------------------------------------------------------

def test_value_at_first_publication(cfnai_alfred):
    """On 2020-02-28 only Jan initial (0.10) is known."""
    result = build_realtime_series(cfnai_alfred)
    row = result[
        (result["series_code"] == "CFNAI") &
        (result["date"] == pd.Timestamp("2020-02-28"))
    ]
    assert len(row) == 1
    assert row.iloc[0]["value"] == pytest.approx(0.10)


def test_value_at_second_publication(cfnai_alfred):
    """On 2020-03-30: Jan revision (0.12, obs=Jan) AND Feb initial (-0.05, obs=Feb).
    Feb is a more recent obs_date so it wins → value=-0.05."""
    result = build_realtime_series(cfnai_alfred)
    row = result[
        (result["series_code"] == "CFNAI") &
        (result["date"] == pd.Timestamp("2020-03-30"))
    ]
    assert len(row) == 1
    assert row.iloc[0]["value"] == pytest.approx(-0.05)


def test_value_at_third_publication(cfnai_alfred):
    """On 2020-04-28: Feb revision (-0.04) is the latest available."""
    result = build_realtime_series(cfnai_alfred)
    row = result[
        (result["series_code"] == "CFNAI") &
        (result["date"] == pd.Timestamp("2020-04-28"))
    ]
    assert len(row) == 1
    assert row.iloc[0]["value"] == pytest.approx(-0.04)


# ---------------------------------------------------------------------------
# Causal integrity — no future values leak in
# ---------------------------------------------------------------------------

def test_no_future_values_at_any_output_row(cfnai_alfred):
    """For every output row, its value must come from a vintage with realtime_start <= row.date."""
    result = build_realtime_series(cfnai_alfred)

    # Reconstruct which value was valid on each pub_date by cross-checking the raw ALFRED data
    for _, row in result.iterrows():
        pub_date = row["date"]
        code = row["series_code"]
        causal = cfnai_alfred[
            (cfnai_alfred["series_code"] == code) &
            (cfnai_alfred["realtime_start"] <= pub_date)
        ]
        assert not causal.empty, f"No causal data for {code} at {pub_date}"
        # row value must be one of the causally available values
        causal_values = set(causal["value"].tolist())
        assert row["value"] in causal_values, (
            f"{code} at {pub_date}: value={row['value']} not in causal set {causal_values}"
        )


# ---------------------------------------------------------------------------
# Deduplication — one row per (series_code, pub_date)
# ---------------------------------------------------------------------------

def test_one_row_per_pub_date_per_series(cfnai_alfred):
    """No duplicate (series_code, date) pairs in output."""
    result = build_realtime_series(cfnai_alfred)
    dup_mask = result.duplicated(subset=["series_code", "date"])
    assert not dup_mask.any(), (
        f"Duplicate (series_code, date) rows found:\n{result[dup_mask]}"
    )


# ---------------------------------------------------------------------------
# series_codes filter
# ---------------------------------------------------------------------------

def test_series_codes_filter_includes_only_requested(two_series_alfred):
    """series_codes=['CFNAI'] → only CFNAI in output."""
    result = build_realtime_series(two_series_alfred, series_codes=["CFNAI"])
    assert set(result["series_code"].unique()) == {"CFNAI"}


def test_series_codes_filter_excludes_others(two_series_alfred):
    """INDPRO must not appear when filtered to CFNAI only."""
    result = build_realtime_series(two_series_alfred, series_codes=["CFNAI"])
    assert "INDPRO" not in result["series_code"].values


def test_series_codes_none_returns_all_series(two_series_alfred):
    """series_codes=None → all series returned."""
    result = build_realtime_series(two_series_alfred, series_codes=None)
    assert set(result["series_code"].unique()) == {"CFNAI", "INDPRO"}


def test_series_codes_nonexistent_raises(cfnai_alfred):
    """Filtering to a series that doesn't exist raises ValueError."""
    with pytest.raises(ValueError):
        build_realtime_series(cfnai_alfred, series_codes=["NONEXISTENT"])


# ---------------------------------------------------------------------------
# Row count = number of unique realtime_start dates per series
# ---------------------------------------------------------------------------

def test_output_row_count_matches_unique_pub_dates(cfnai_alfred):
    """CFNAI has 3 unique realtime_start dates → 3 output rows."""
    result = build_realtime_series(cfnai_alfred)
    cfnai_rows = result[result["series_code"] == "CFNAI"]
    unique_pub_dates = cfnai_alfred["realtime_start"].nunique()
    assert len(cfnai_rows) == unique_pub_dates


# ---------------------------------------------------------------------------
# Output is a drop-in FRED replacement (dtype / schema)
# ---------------------------------------------------------------------------

def test_output_value_dtype_is_numeric(cfnai_alfred):
    """Output 'value' column must be numeric."""
    result = build_realtime_series(cfnai_alfred)
    assert pd.api.types.is_numeric_dtype(result["value"])


def test_output_date_dtype_is_datetime(cfnai_alfred):
    """Output 'date' column must be datetime."""
    result = build_realtime_series(cfnai_alfred)
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


# ---------------------------------------------------------------------------
# YAML config — use_alfred defaults to false
# ---------------------------------------------------------------------------

def test_yaml_use_alfred_defaults_to_false():
    """configs/data/regime_universe.yaml must have alfred.use_alfred=false."""
    import yaml
    from pathlib import Path
    config_path = Path(__file__).parent.parent / "configs" / "data" / "regime_universe.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    alfred_cfg = cfg.get("regime_universe", {}).get("alfred", {})
    assert alfred_cfg.get("use_alfred") is False, (
        "alfred.use_alfred should default to false in YAML"
    )
