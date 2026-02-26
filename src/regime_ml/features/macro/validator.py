"""
Macro Feature Validator

Validates generated macro features for statistical validity, data quality,
and correct behavior of transforms. Catches common issues like:
- Features computed on forward-filled data (artificially low variance)
- Missing value patterns indicating burn-in issues
- Z-score distributions that don't match expected properties
- Different frequency indicators showing inappropriate variation
- Index alignment issues

Frequency Detection:
    Indicator frequencies (daily/weekly/monthly) are automatically derived from
    the YAML configuration 'frequency' field. This eliminates manual specification
    and ensures consistency with data source definitions.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results."""
    passed: bool
    message: str
    severity: str  # 'info', 'warning', 'error'
    details: Optional[Dict] = None


class MacroFeatureValidator:
    """
    Validates macro features for statistical properties and data quality.

    Performs checks on:
    - Feature completeness and missing values
    - Statistical distributions (z-scores, outliers)
    - Temporal consistency
    - Daily/weekly/monthly indicator behavior
    - Index properties
    """

    def __init__(
        self,
        feature_data: pd.DataFrame,
        frequency_map: Optional[Dict[str, str]] = None
    ):
        """
        Initialize validator.

        Args:
            feature_data: DataFrame of generated features
            frequency_map: Dict mapping series_code to frequency ('daily', 'weekly', 'monthly')
                          If None, uses default categorization
        """
        self.df = feature_data

        # Use provided frequency map or defaults
        if frequency_map is not None:
            self.frequency_map = frequency_map
        else:
            # Default fallback if no frequency map provided
            self.frequency_map = {
                'VIXCLS': 'daily',
                'DGS2': 'daily',
                'DGS10': 'daily',
                'T10Y3M': 'daily',
                'NFCI': 'weekly',
                'ICSA': 'weekly',
                'CFNAI': 'monthly',
                'INDPRO': 'monthly',
                'PCEPILFE': 'monthly'
            }

        # Categorize by frequency
        self.daily_indicators = [k for k, v in self.frequency_map.items() if v == 'daily']
        self.weekly_indicators = [k for k, v in self.frequency_map.items() if v == 'weekly']
        self.monthly_indicators = [k for k, v in self.frequency_map.items() if v == 'monthly']

        self.results: List[ValidationResult] = []

    def validate_all(self) -> List[ValidationResult]:
        """
        Run all validation checks.

        Returns:
            List of ValidationResult objects
        """
        logger.info("=" * 80)
        logger.info("MACRO FEATURE VALIDATION")
        logger.info("=" * 80)

        # Run all checks
        self._check_index_properties()
        self._check_missing_values()
        self._check_zscore_distributions()
        self._check_outliers()
        self._check_monthly_indicators()
        self._check_weekly_indicators()
        self._check_daily_indicators()
        self._check_feature_variance()
        self._check_temporal_consistency()
        self._check_feature_correlations()
        self._check_stationarity()

        # Summarize results
        self._log_summary()

        return self.results

    def _check_index_properties(self) -> None:
        """Validate index properties."""
        logger.info("\n1. Index Properties")
        logger.info("-" * 40)

        # Check index type
        if isinstance(self.df.index, pd.DatetimeIndex):
            self.results.append(ValidationResult(
                passed=True,
                message="Index is DatetimeIndex",
                severity='info',
                details={'type': str(type(self.df.index))}
            ))
            logger.info("[PASS] Index is DatetimeIndex")
        else:
            self.results.append(ValidationResult(
                passed=False,
                message="Index is not DatetimeIndex: %s" % type(self.df.index),
                severity='error'
            ))
            logger.warning("[FAIL] Index type: %s", type(self.df.index))

        # Check for duplicates
        if not self.df.index.duplicated().any():
            self.results.append(ValidationResult(
                passed=True,
                message="No duplicate dates in index",
                severity='info'
            ))
            logger.info("[PASS] No duplicate dates")
        else:
            n_dupes = self.df.index.duplicated().sum()
            self.results.append(ValidationResult(
                passed=False,
                message="Found %d duplicate dates" % n_dupes,
                severity='error',
                details={'n_duplicates': n_dupes}
            ))
            logger.warning("[FAIL] Found %d duplicate dates", n_dupes)

        # Check if sorted
        if self.df.index.is_monotonic_increasing:
            self.results.append(ValidationResult(
                passed=True,
                message="Index is sorted",
                severity='info'
            ))
            logger.info("[PASS] Index is sorted")
        else:
            self.results.append(ValidationResult(
                passed=False,
                message="Index is not sorted",
                severity='warning'
            ))
            logger.warning("[WARN] Index is not sorted")

        # Check frequency against master calendar (US business days)
        if isinstance(self.df.index, pd.DatetimeIndex) and len(self.df.index) > 0:
            try:
                # Create US business day calendar
                us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

                # Generate expected business day calendar for the date range
                expected_calendar = pd.date_range(
                    start=self.df.index.min(),
                    end=self.df.index.max(),
                    freq=us_bd
                )

                # Check alignment with master calendar
                index_set = set(self.df.index.normalize())  # type: ignore[attr-defined]  # DatetimeIndex.normalize() not in pandas stubs for all Index types
                expected_set = set(expected_calendar.normalize())  # type: ignore[attr-defined]  # DatetimeIndex.normalize() not in pandas stubs for all Index types

                missing_dates = expected_set - index_set
                extra_dates = index_set - expected_set

                if len(missing_dates) == 0 and len(extra_dates) == 0:
                    self.results.append(ValidationResult(
                        passed=True,
                        message="Index aligns with US business day calendar",
                        severity='info'
                    ))
                    logger.info("[PASS] Frequency: US business days (USFederalHolidayCalendar)")
                else:
                    details = {}
                    if len(missing_dates) > 0:
                        details['missing_dates'] = len(missing_dates)
                    if len(extra_dates) > 0:
                        details['extra_dates'] = len(extra_dates)

                    self.results.append(ValidationResult(
                        passed=False,
                        message="Index does not align with US business day calendar",
                        severity='warning',
                        details=details
                    ))

                    if len(missing_dates) > 0:
                        logger.warning("[WARN] Missing %d expected business days", len(missing_dates))
                    if len(extra_dates) > 0:
                        logger.warning("[WARN] Contains %d non-business days", len(extra_dates))

            except Exception as e:
                self.results.append(ValidationResult(
                    passed=False,
                    message="Could not validate calendar alignment: %s" % str(e),
                    severity='warning'
                ))
                logger.warning("[WARN] Calendar validation failed: %s", str(e))
        else:
            logger.warning("[WARN] Cannot validate frequency (empty or invalid index)")

    def _check_missing_values(self) -> None:
        """Check missing value patterns."""
        logger.info("\n2. Missing Values")
        logger.info("-" * 40)

        total_nulls = self.df.isnull().sum().sum()
        total_values = self.df.size
        pct_null = 100 * total_nulls / total_values

        logger.info("Total nulls: %d / %d (%.2f%%)", total_nulls, total_values, pct_null)

        # Check for features with high null percentage
        null_pct_by_col = 100 * self.df.isnull().sum() / len(self.df)
        high_null_cols = null_pct_by_col[null_pct_by_col > 20]

        if len(high_null_cols) > 0:
            self.results.append(ValidationResult(
                passed=False,
                message="%d features have >20%% nulls" % len(high_null_cols),
                severity='error',
                details={'features': high_null_cols.to_dict()}
            ))
            logger.warning("[FAIL] %d features with >20%% nulls:", len(high_null_cols))
            for col, pct in high_null_cols.items():
                logger.warning("       %s: %.1f%%", col, pct)
        else:
            self.results.append(ValidationResult(
                passed=True,
                message="No features with >20% nulls",
                severity='info'
            ))
            logger.info("[PASS] No features with >20%% nulls")

        # Check early burn-in period
        if len(self.df) > 0:
            df_clean = self.df.dropna()
            if len(df_clean) > 0:
                first_complete_idx = df_clean.index.min()
                first_idx = self.df.index.min()

                if pd.notna(first_complete_idx) and pd.notna(first_idx):  # type: ignore[arg-type]  # pd.notna accepts Index scalars; mypy does not narrow from Index.min()
                    try:
                        burn_in_timedelta = first_complete_idx - first_idx  # type: ignore[operator]  # Timestamp subtraction; mypy cannot infer return type from Index.min()
                        burn_in_days = burn_in_timedelta.days if hasattr(burn_in_timedelta, 'days') else 0  # type: ignore[union-attr]  # timedelta .days access; union-typed after subtraction
                        burn_in_years = burn_in_days / 365.25

                        logger.info("  First complete row: %s", first_complete_idx)
                        logger.info("  Burn-in period: %d days (%.1f years)", burn_in_days, burn_in_years)

                        if burn_in_years > 10:
                            self.results.append(ValidationResult(
                                passed=False,
                                message="Burn-in period too long: %.1f years" % burn_in_years,
                                severity='warning',
                                details={'burn_in_years': burn_in_years}
                            ))
                        else:
                            self.results.append(ValidationResult(
                                passed=True,
                                message="Burn-in period acceptable: %.1f years" % burn_in_years,
                                severity='info'
                            ))
                    except (TypeError, AttributeError):
                        pass  # Skip burn-in calculation if types don't match

    def _check_zscore_distributions(self) -> None:
        """Check if z-score features have expected properties."""
        logger.info("\n3. Z-Score Distributions")
        logger.info("-" * 40)

        zscore_cols = [col for col in self.df.columns if 'zscore' in col.lower()]

        if len(zscore_cols) == 0:
            logger.info("  No z-score features found")
            return

        logger.info("  Checking %d z-score features...", len(zscore_cols))

        issues = []
        for col in zscore_cols:
            series = self.df[col].dropna()
            if len(series) < 100:
                continue

            mean = series.mean()
            std = series.std()

            # Z-scores should have mean ≈ 0 and std ≈ 1
            if abs(mean) > 0.5:
                issues.append("%s: mean=%.2f (expected ≈0)" % (col, mean))

            if abs(std - 1.0) > 0.5:
                issues.append("%s: std=%.2f (expected ≈1)" % (col, std))

        if len(issues) > 0:
            self.results.append(ValidationResult(
                passed=False,
                message="%d z-score features have unusual distributions" % len(issues),
                severity='warning',
                details={'issues': issues[:5]}
            ))
            logger.warning("[WARN] %d features with unusual distributions:", len(issues))
            for issue in issues[:5]:
                logger.warning("       %s", issue)
            if len(issues) > 5:
                logger.warning("       ... and %d more", len(issues) - 5)
        else:
            self.results.append(ValidationResult(
                passed=True,
                message="All z-score distributions look normal",
                severity='info'
            ))
            logger.info("[PASS] Z-score distributions look normal")

    def _check_outliers(self) -> None:
        """Check for extreme outliers."""
        logger.info("\n4. Outliers")
        logger.info("-" * 40)

        # Check for extreme values (|z| > 10)
        extreme_threshold = 10

        extreme_counts = {}
        for col in self.df.columns:
            n_extreme = (self.df[col].abs() > extreme_threshold).sum()
            if n_extreme > 0:
                extreme_counts[col] = n_extreme

        if len(extreme_counts) > 0:
            total_extreme = sum(extreme_counts.values())
            self.results.append(ValidationResult(
                passed=False,
                message="Found %d extreme values (|value| > %d)" % (total_extreme, extreme_threshold),
                severity='warning',
                details={'features': extreme_counts}
            ))
            logger.warning("[WARN] Features with extreme values (|value| > %d):", extreme_threshold)
            for col, count in list(extreme_counts.items())[:5]:
                pct = 100 * count / self.df[col].notna().sum()
                logger.warning("       %s: %d values (%.2f%%)", col, count, pct)
            if len(extreme_counts) > 5:
                logger.warning("       ... and %d more features", len(extreme_counts) - 5)
        else:
            self.results.append(ValidationResult(
                passed=True,
                message="No extreme outliers (|value| > %d)" % extreme_threshold,
                severity='info'
            ))
            logger.info("[PASS] No extreme outliers")

    def _check_monthly_indicators(self) -> None:
        """Check if monthly indicators show step-function behavior."""
        logger.info("\n5. Monthly Indicators")
        logger.info("-" * 40)

        monthly_features = [col for col in self.df.columns
                           if any(ind in col for ind in self.monthly_indicators)]

        if len(monthly_features) == 0:
            logger.info("  No monthly indicator features found")
            return

        logger.info("  Checking %d monthly features...", len(monthly_features))

        issues = []
        for col in monthly_features:
            # Check if feature changes daily (should be step function)
            series = self.df[col].dropna()
            if len(series) < 100:
                continue

            # Count consecutive days with same value
            changes = (series != series.shift()).sum()
            change_rate = changes / len(series)

            # Monthly data should change ~21 times per year = ~0.083 change rate
            # Daily data changes ~252 times per year = ~1.0 change rate
            if change_rate > 0.15:  # More than twice expected for monthly
                issues.append("%s: changes %d times (%.2f%% rate)" % (col, changes, change_rate * 100))

        if len(issues) > 0:
            self.results.append(ValidationResult(
                passed=False,
                message="%d monthly features show daily variation" % len(issues),
                severity='error',
                details={'issues': issues[:3]}
            ))
            logger.warning("[FAIL] %d features show daily variation (expected monthly steps):", len(issues))
            for issue in issues[:3]:
                logger.warning("       %s", issue)
        else:
            self.results.append(ValidationResult(
                passed=True,
                message="Monthly features show step-function behavior",
                severity='info'
            ))
            logger.info("[PASS] Monthly features show expected step-function behavior")

    def _check_weekly_indicators(self) -> None:
        """Check if weekly indicators show appropriate step-function behavior."""
        logger.info("\n6. Weekly Indicators")
        logger.info("-" * 40)

        weekly_features = [col for col in self.df.columns
                          if any(ind in col for ind in self.weekly_indicators)]

        if len(weekly_features) == 0:
            logger.info("  No weekly indicator features found")
            return

        logger.info("  Checking %d weekly features...", len(weekly_features))
        logger.debug("  Weekly indicators: %s", ", ".join(self.weekly_indicators))

        issues = []
        for col in weekly_features:
            # Check if feature changes at appropriate rate
            series = self.df[col].dropna()
            if len(series) < 100:
                continue

            # Count changes
            changes = (series != series.shift()).sum()
            change_rate = changes / len(series)

            # Weekly data should change ~52 times per year = ~0.21 change rate
            # Allow range of 0.10 to 0.35 for weekly
            if change_rate < 0.10 or change_rate > 0.35:
                issues.append("%s: changes %d times (%.2f%% rate)" % (col, changes, change_rate * 100))

        if len(issues) > 0:
            self.results.append(ValidationResult(
                passed=False,
                message="%d weekly features show unexpected variation" % len(issues),
                severity='warning',
                details={'issues': issues[:3]}
            ))
            logger.warning("[WARN] %d features show unexpected variation:", len(issues))
            for issue in issues[:3]:
                logger.warning("       %s", issue)
        else:
            self.results.append(ValidationResult(
                passed=True,
                message="Weekly features show expected step-function behavior",
                severity='info'
            ))
            logger.info("[PASS] Weekly features show expected step-function behavior")

    def _check_daily_indicators(self) -> None:
        """Check if daily indicators show daily variation."""
        logger.info("\n7. Daily Indicators")
        logger.info("-" * 40)

        daily_features = [col for col in self.df.columns
                         if any(ind in col for ind in self.daily_indicators)]

        if len(daily_features) == 0:
            logger.info("  No daily indicator features found")
            return

        logger.info("  Checking %d daily features...", len(daily_features))

        issues = []
        for col in daily_features:
            series = self.df[col].dropna()
            if len(series) < 100:
                continue

            # Daily features should change frequently
            changes = (series != series.shift()).sum()
            change_rate = changes / len(series)

            # Daily data should have high change rate (>0.5)
            if change_rate < 0.5:
                issues.append("%s: only changes %d times (%.2f%% rate)" % (col, changes, change_rate * 100))

        if len(issues) > 0:
            self.results.append(ValidationResult(
                passed=False,
                message="%d daily features show low variation" % len(issues),
                severity='warning',
                details={'issues': issues[:3]}
            ))
            logger.warning("[WARN] %d features show low variation:", len(issues))
            for issue in issues[:3]:
                logger.warning("       %s", issue)
        else:
            self.results.append(ValidationResult(
                passed=True,
                message="Daily features show expected variation",
                severity='info'
            ))
            logger.info("[PASS] Daily features show expected variation")

    def _check_feature_variance(self) -> None:
        """Check for suspiciously low variance features."""
        logger.info("\n8. Feature Variance")
        logger.info("-" * 40)

        # Check for features with very low variance
        low_var_threshold = 0.01
        variances = self.df.var()

        if isinstance(variances, pd.Series):
            low_var_features = variances[variances < low_var_threshold]

            if isinstance(low_var_features, pd.Series) and len(low_var_features) > 0:
                low_var_dict = dict(low_var_features)
                self.results.append(ValidationResult(
                    passed=False,
                    message="%d features have very low variance (<%.2f)" % (len(low_var_features), low_var_threshold),
                    severity='warning',
                    details={'features': low_var_dict}
                ))
                logger.warning("[WARN] %d features with very low variance:", len(low_var_features))
                for col, var in list(low_var_dict.items())[:5]:
                    logger.warning("       %s: variance=%.6f", col, var)
            else:
                logger.info("[PASS] All features have reasonable variance")
        else:
            logger.warning("[WARN] Could not compute variance")

    def _check_temporal_consistency(self) -> None:
        """Check for temporal consistency issues."""
        logger.info("\n9. Temporal Consistency")
        logger.info("-" * 40)

        # Check for features that suddenly jump or have discontinuities
        issues = []

        for col in self.df.columns:
            series = self.df[col].dropna()
            if len(series) < 100:
                continue

            # Check for very large jumps (>5 std devs)
            diffs = series.diff().abs()
            if len(diffs) > 0 and diffs.std() > 0:
                threshold = diffs.mean() + 5 * diffs.std()
                large_jumps = (diffs > threshold).sum()

                if large_jumps > len(series) * 0.01:  # More than 1% jumps
                    issues.append("%s: %d large jumps" % (col, large_jumps))

        if len(issues) > 0:
            self.results.append(ValidationResult(
                passed=False,
                message="%d features have discontinuities" % len(issues),
                severity='warning',
                details={'issues': issues[:3]}
            ))
            logger.warning("[WARN] %d features with potential discontinuities:", len(issues))
            for issue in issues[:3]:
                logger.warning("       %s", issue)
        else:
            self.results.append(ValidationResult(
                passed=True,
                message="No major temporal discontinuities",
                severity='info'
            ))
            logger.info("[PASS] No major temporal discontinuities")

    def _check_feature_correlations(self) -> None:
        """Check for high feature correlations."""
        logger.info("\n10. Feature Correlations")
        logger.info("-" * 40)

        # Correlation threshold
        threshold = 0.70

        # Calculate correlation matrix
        corr_matrix = self.df.corr().abs()

        # Find pairs with correlation > threshold
        # Use upper triangle to avoid duplicates
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if corr_value > threshold:
                    high_corr_pairs.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })

        # Get max correlation (excluding diagonal)
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        max_corr = upper_triangle.max().max()

        logger.info(
            "  Total feature pairs: %d",
            len(corr_matrix.columns) * (len(corr_matrix.columns) - 1) // 2
        )
        logger.info("  Maximum correlation: %.3f", max_corr)
        logger.info("  Threshold: %.2f", threshold)

        if len(high_corr_pairs) > 0:
            # Sort by correlation (descending)
            high_corr_pairs = sorted(high_corr_pairs,
                                     key=lambda x: x['correlation'],
                                     reverse=True)

            self.results.append(ValidationResult(
                passed=False,
                message="%d feature pairs with correlation >%.2f" % (len(high_corr_pairs), threshold),
                severity='warning',
                details={'high_corr_pairs': high_corr_pairs[:5]}
            ))
            logger.warning("[WARN] Found %d feature pairs with correlation >%.2f:", len(high_corr_pairs), threshold)
            for pair in high_corr_pairs[:5]:
                logger.warning(
                    "       %s <-> %s: %.3f",
                    pair['feature_1'], pair['feature_2'], pair['correlation']
                )
            if len(high_corr_pairs) > 5:
                logger.warning("       ... and %d more pairs", len(high_corr_pairs) - 5)
        else:
            self.results.append(ValidationResult(
                passed=True,
                message="No feature pairs with correlation >%.2f" % threshold,
                severity='info'
            ))
            logger.info("[PASS] All feature correlations <%.2f", threshold)

    def _check_stationarity(self) -> None:
        """Check features for stationarity using Augmented Dickey-Fuller test."""
        logger.info("\n11. Stationarity (ADF Test)")
        logger.info("-" * 40)

        # Significance level
        alpha = 0.05

        # Test each feature
        non_stationary = []
        stationary_count = 0

        for col in self.df.columns:
            try:
                # Drop NaN values for ADF test
                series = self.df[col].dropna()

                if len(series) < 50:  # Need sufficient data for ADF test
                    continue

                # Run ADF test
                adf_result = adfuller(series, autolag='AIC')
                p_value = adf_result[1]

                if p_value >= alpha:
                    # Non-stationary (cannot reject null hypothesis)
                    non_stationary.append({
                        'feature': col,
                        'p_value': p_value,
                        'adf_stat': adf_result[0]
                    })
                else:
                    stationary_count += 1

            except Exception as e:
                # Skip features that cause errors in ADF test
                logger.debug("[WARN] Could not test %s: %s", col, str(e)[:50])
                continue

        total_tested = stationary_count + len(non_stationary)

        logger.info("  Features tested: %d", total_tested)
        logger.info("  Stationary (p < %.2f): %d", alpha, stationary_count)
        logger.info("  Non-stationary (p >= %.2f): %d", alpha, len(non_stationary))

        if len(non_stationary) > 0:
            # Sort by p-value (most non-stationary first)
            non_stationary = sorted(non_stationary,
                                    key=lambda x: x['p_value'],
                                    reverse=True)

            self.results.append(ValidationResult(
                passed=False,
                message="%d features are non-stationary (p >= %.2f)" % (len(non_stationary), alpha),
                severity='warning',
                details={'non_stationary': non_stationary[:5]}
            ))
            logger.warning("[WARN] %d non-stationary features (may have trends/unit roots):", len(non_stationary))
            for item in non_stationary[:5]:
                logger.warning(
                    "       %s: p=%.4f, ADF=%.3f",
                    item['feature'], item['p_value'], item['adf_stat']
                )
            if len(non_stationary) > 5:
                logger.warning("       ... and %d more", len(non_stationary) - 5)
            logger.info("       Note: Non-stationary features may need differencing or detrending")
        else:
            self.results.append(ValidationResult(
                passed=True,
                message="All features are stationary (p < %.2f)" % alpha,
                severity='info'
            ))
            logger.info("[PASS] All %d features are stationary", total_tested)

    def _log_summary(self) -> None:
        """Log validation summary."""
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)

        errors = [r for r in self.results if r.severity == 'error' and not r.passed]
        warnings = [r for r in self.results if r.severity == 'warning' and not r.passed]
        passed = [r for r in self.results if r.passed]

        logger.info("[PASS] Passed: %d", len(passed))
        logger.info("[WARN] Warnings: %d", len(warnings))
        logger.info("[FAIL] Errors: %d", len(errors))

        if errors:
            logger.warning("ERRORS:")
            for r in errors:
                logger.warning("  - %s", r.message)

        if warnings:
            logger.warning("WARNINGS:")
            for r in warnings[:5]:
                logger.warning("  - %s", r.message)
            if len(warnings) > 5:
                logger.warning("  - ... and %d more warnings", len(warnings) - 5)

        logger.info("=" * 80)


def validate_macro_features(
    feature_data: pd.DataFrame,
    frequency_map: Optional[Dict[str, str]] = None
) -> List[ValidationResult]:
    """
    Convenience function to validate macro features.

    Args:
        feature_data: DataFrame of generated features
        frequency_map: Dict mapping series_code to frequency ('daily', 'weekly', 'monthly')

    Returns:
        List of ValidationResult objects
    """
    validator = MacroFeatureValidator(feature_data, frequency_map)
    return validator.validate_all()
