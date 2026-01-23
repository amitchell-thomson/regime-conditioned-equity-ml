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

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from statsmodels.tsa.stattools import adfuller


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
        print("="*80)
        print("MACRO FEATURE VALIDATION")
        print("="*80)
        
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
        self._print_summary()
        
        return self.results
    
    def _check_index_properties(self):
        """Validate index properties."""
        print("\n1. Index Properties")
        print("-" * 40)
        
        # Check index type
        if isinstance(self.df.index, pd.DatetimeIndex):
            self.results.append(ValidationResult(
                passed=True,
                message="Index is DatetimeIndex",
                severity='info',
                details={'type': str(type(self.df.index))}
            ))
            print("[PASS] Index is DatetimeIndex")
        else:
            self.results.append(ValidationResult(
                passed=False,
                message=f"Index is not DatetimeIndex: {type(self.df.index)}",
                severity='error'
            ))
            print(f"[FAIL] Index type: {type(self.df.index)}")
        
        # Check for duplicates
        if not self.df.index.duplicated().any():
            self.results.append(ValidationResult(
                passed=True,
                message="No duplicate dates in index",
                severity='info'
            ))
            print("[PASS] No duplicate dates")
        else:
            n_dupes = self.df.index.duplicated().sum()
            self.results.append(ValidationResult(
                passed=False,
                message=f"Found {n_dupes} duplicate dates",
                severity='error',
                details={'n_duplicates': n_dupes}
            ))
            print(f"[FAIL] Found {n_dupes} duplicate dates")
        
        # Check if sorted
        if self.df.index.is_monotonic_increasing:
            self.results.append(ValidationResult(
                passed=True,
                message="Index is sorted",
                severity='info'
            ))
            print("[PASS] Index is sorted")
        else:
            self.results.append(ValidationResult(
                passed=False,
                message="Index is not sorted",
                severity='warning'
            ))
            print("[WARN] Index is not sorted")
        
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
                index_set = set(self.df.index.normalize()) # type: ignore
                expected_set = set(expected_calendar.normalize()) # type: ignore
                
                missing_dates = expected_set - index_set
                extra_dates = index_set - expected_set
                
                if len(missing_dates) == 0 and len(extra_dates) == 0:
                    self.results.append(ValidationResult(
                        passed=True,
                        message="Index aligns with US business day calendar",
                        severity='info'
                    ))
                    print("[PASS] Frequency: US business days (USFederalHolidayCalendar)")
                else:
                    details = {}
                    if len(missing_dates) > 0:
                        details['missing_dates'] = len(missing_dates)
                    if len(extra_dates) > 0:
                        details['extra_dates'] = len(extra_dates)
                    
                    self.results.append(ValidationResult(
                        passed=False,
                        message=f"Index does not align with US business day calendar",
                        severity='warning',
                        details=details
                    ))
                    
                    if len(missing_dates) > 0:
                        print(f"[WARN] Missing {len(missing_dates)} expected business days")
                    if len(extra_dates) > 0:
                        print(f"[WARN] Contains {len(extra_dates)} non-business days")
                    
            except Exception as e:
                self.results.append(ValidationResult(
                    passed=False,
                    message=f"Could not validate calendar alignment: {str(e)}",
                    severity='warning'
                ))
                print(f"[WARN] Calendar validation failed: {str(e)}")
        else:
            print("[WARN] Cannot validate frequency (empty or invalid index)")
    
    def _check_missing_values(self):
        """Check missing value patterns."""
        print("\n2. Missing Values")
        print("-" * 40)
        
        total_nulls = self.df.isnull().sum().sum()
        total_values = self.df.size
        pct_null = 100 * total_nulls / total_values
        
        print(f"Total nulls: {total_nulls:,} / {total_values:,} ({pct_null:.2f}%)")
        
        # Check for features with high null percentage
        null_pct_by_col = 100 * self.df.isnull().sum() / len(self.df)
        high_null_cols = null_pct_by_col[null_pct_by_col > 20]
        
        if len(high_null_cols) > 0:
            self.results.append(ValidationResult(
                passed=False,
                message=f"{len(high_null_cols)} features have >20% nulls",
                severity='error',
                details={'features': high_null_cols.to_dict()}
            ))
            print(f"[FAIL] {len(high_null_cols)} features with >20% nulls:")
            for col, pct in high_null_cols.items():
                print(f"       {col}: {pct:.1f}%")
        else:
            self.results.append(ValidationResult(
                passed=True,
                message="No features with >20% nulls",
                severity='info'
            ))
            print("[PASS] No features with >20% nulls")
        
        # Check early burn-in period
        if len(self.df) > 0:
            df_clean = self.df.dropna()
            if len(df_clean) > 0:
                first_complete_idx = df_clean.index.min()
                first_idx = self.df.index.min()
                
                if pd.notna(first_complete_idx) and pd.notna(first_idx): # type: ignore
                    try:
                        burn_in_timedelta = first_complete_idx - first_idx  # type: ignore
                        burn_in_days = burn_in_timedelta.days if hasattr(burn_in_timedelta, 'days') else 0 # type: ignore
                        burn_in_years = burn_in_days / 365.25
                        
                        print(f"  First complete row: {first_complete_idx}")
                        print(f"  Burn-in period: {burn_in_days} days ({burn_in_years:.1f} years)")
                        
                        if burn_in_years > 10:
                            self.results.append(ValidationResult(
                                passed=False,
                                message=f"Burn-in period too long: {burn_in_years:.1f} years",
                                severity='warning',
                                details={'burn_in_years': burn_in_years}
                            ))
                        else:
                            self.results.append(ValidationResult(
                                passed=True,
                                message=f"Burn-in period acceptable: {burn_in_years:.1f} years",
                                severity='info'
                            ))
                    except (TypeError, AttributeError):
                        pass  # Skip burn-in calculation if types don't match
    
    def _check_zscore_distributions(self):
        """Check if z-score features have expected properties."""
        print("\n3. Z-Score Distributions")
        print("-" * 40)
        
        zscore_cols = [col for col in self.df.columns if 'zscore' in col.lower()]
        
        if len(zscore_cols) == 0:
            print("  No z-score features found")
            return
        
        print(f"  Checking {len(zscore_cols)} z-score features...")
        
        issues = []
        for col in zscore_cols:
            series = self.df[col].dropna()
            if len(series) < 100:
                continue
            
            mean = series.mean()
            std = series.std()
            
            # Z-scores should have mean ≈ 0 and std ≈ 1
            if abs(mean) > 0.5:
                issues.append(f"{col}: mean={mean:.2f} (expected ≈0)")
            
            if abs(std - 1.0) > 0.5:
                issues.append(f"{col}: std={std:.2f} (expected ≈1)")
        
        if len(issues) > 0:
            self.results.append(ValidationResult(
                passed=False,
                message=f"{len(issues)} z-score features have unusual distributions",
                severity='warning',
                details={'issues': issues[:5]}  # First 5 issues
            ))
            print(f"[WARN] {len(issues)} features with unusual distributions:")
            for issue in issues[:5]:
                print(f"       {issue}")
            if len(issues) > 5:
                print(f"       ... and {len(issues) - 5} more")
        else:
            self.results.append(ValidationResult(
                passed=True,
                message="All z-score distributions look normal",
                severity='info'
            ))
            print("[PASS] Z-score distributions look normal")
    
    def _check_outliers(self):
        """Check for extreme outliers."""
        print("\n4. Outliers")
        print("-" * 40)
        
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
                message=f"Found {total_extreme} extreme values (|value| > {extreme_threshold})",
                severity='warning',
                details={'features': extreme_counts}
            ))
            print(f"[WARN] Features with extreme values (|value| > {extreme_threshold}):")
            for col, count in list(extreme_counts.items())[:5]:
                pct = 100 * count / self.df[col].notna().sum()
                print(f"       {col}: {count} values ({pct:.2f}%)")
            if len(extreme_counts) > 5:
                print(f"       ... and {len(extreme_counts) - 5} more features")
        else:
            self.results.append(ValidationResult(
                passed=True,
                message=f"No extreme outliers (|value| > {extreme_threshold})",
                severity='info'
            ))
            print(f"[PASS] No extreme outliers")
    
    def _check_monthly_indicators(self):
        """Check if monthly indicators show step-function behavior."""
        print("\n5. Monthly Indicators")
        print("-" * 40)
        
        monthly_features = [col for col in self.df.columns 
                           if any(ind in col for ind in self.monthly_indicators)]
        
        if len(monthly_features) == 0:
            print("  No monthly indicator features found")
            return
        
        print(f"  Checking {len(monthly_features)} monthly features...")
        
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
                issues.append(f"{col}: changes {changes} times ({change_rate:.2%} rate)")
        
        if len(issues) > 0:
            self.results.append(ValidationResult(
                passed=False,
                message=f"{len(issues)} monthly features show daily variation",
                severity='error',
                details={'issues': issues[:3]}
            ))
            print(f"[FAIL] {len(issues)} features show daily variation (expected monthly steps):")
            for issue in issues[:3]:
                print(f"       {issue}")
        else:
            self.results.append(ValidationResult(
                passed=True,
                message="Monthly features show step-function behavior",
                severity='info'
            ))
            print("[PASS] Monthly features show expected step-function behavior")
    
    def _check_weekly_indicators(self):
        """Check if weekly indicators show appropriate step-function behavior."""
        print("\n6. Weekly Indicators")
        print("-" * 40)
        
        weekly_features = [col for col in self.df.columns 
                          if any(ind in col for ind in self.weekly_indicators)]
        
        if len(weekly_features) == 0:
            print("  No weekly indicator features found")
            return
        
        print(f"  Checking {len(weekly_features)} weekly features...")
        print(f"  Weekly indicators: {', '.join(self.weekly_indicators)}")
        
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
                issues.append(f"{col}: changes {changes} times ({change_rate:.2%} rate)")
        
        if len(issues) > 0:
            self.results.append(ValidationResult(
                passed=False,
                message=f"{len(issues)} weekly features show unexpected variation",
                severity='warning',
                details={'issues': issues[:3]}
            ))
            print(f"[WARN] {len(issues)} features show unexpected variation:")
            for issue in issues[:3]:
                print(f"       {issue}")
        else:
            self.results.append(ValidationResult(
                passed=True,
                message="Weekly features show expected step-function behavior",
                severity='info'
            ))
            print("[PASS] Weekly features show expected step-function behavior")
    
    def _check_daily_indicators(self):
        """Check if daily indicators show daily variation."""
        print("\n7. Daily Indicators")
        print("-" * 40)
        
        daily_features = [col for col in self.df.columns 
                         if any(ind in col for ind in self.daily_indicators)]
        
        if len(daily_features) == 0:
            print("  No daily indicator features found")
            return
        
        print(f"  Checking {len(daily_features)} daily features...")
        
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
                issues.append(f"{col}: only changes {changes} times ({change_rate:.2%} rate)")
        
        if len(issues) > 0:
            self.results.append(ValidationResult(
                passed=False,
                message=f"{len(issues)} daily features show low variation",
                severity='warning',
                details={'issues': issues[:3]}
            ))
            print(f"[WARN] {len(issues)} features show low variation:")
            for issue in issues[:3]:
                print(f"       {issue}")
        else:
            self.results.append(ValidationResult(
                passed=True,
                message="Daily features show expected variation",
                severity='info'
            ))
            print("[PASS] Daily features show expected variation")
    
    def _check_feature_variance(self):
        """Check for suspiciously low variance features."""
        print("\n8. Feature Variance")
        print("-" * 40)
        
        # Check for features with very low variance
        low_var_threshold = 0.01
        variances = self.df.var()
        
        if isinstance(variances, pd.Series):
            low_var_features = variances[variances < low_var_threshold]
            
            if isinstance(low_var_features, pd.Series) and len(low_var_features) > 0:
                low_var_dict = dict(low_var_features)
                self.results.append(ValidationResult(
                    passed=False,
                    message=f"{len(low_var_features)} features have very low variance (<{low_var_threshold})",
                    severity='warning',
                    details={'features': low_var_dict}
                ))
                print(f"[WARN] {len(low_var_features)} features with very low variance:")
                for col, var in list(low_var_dict.items())[:5]:
                    print(f"       {col}: variance={var:.6f}")
            else:
                print("[PASS] All features have reasonable variance")
        else:
            print("[WARN] Could not compute variance")
    
    def _check_temporal_consistency(self):
        """Check for temporal consistency issues."""
        print("\n9. Temporal Consistency")
        print("-" * 40)
        
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
                    issues.append(f"{col}: {large_jumps} large jumps")
        
        if len(issues) > 0:
            self.results.append(ValidationResult(
                passed=False,
                message=f"{len(issues)} features have discontinuities",
                severity='warning',
                details={'issues': issues[:3]}
            ))
            print(f"[WARN] {len(issues)} features with potential discontinuities:")
            for issue in issues[:3]:
                print(f"       {issue}")
        else:
            self.results.append(ValidationResult(
                passed=True,
                message="No major temporal discontinuities",
                severity='info'
            ))
            print("[PASS] No major temporal discontinuities")
    
    def _check_feature_correlations(self):
        """Check for high feature correlations."""
        print("\n10. Feature Correlations")
        print("-" * 40)
        
        # Correlation threshold
        threshold = 0.70
        
        # Calculate correlation matrix
        corr_matrix = self.df.corr().abs()
        
        # Find pairs with correlation > threshold
        # Use upper triangle to avoid duplicates
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
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
        
        print(f"  Total feature pairs: {len(corr_matrix.columns) * (len(corr_matrix.columns) - 1) // 2}")
        print(f"  Maximum correlation: {max_corr:.3f}")
        print(f"  Threshold: {threshold}")
        
        if len(high_corr_pairs) > 0:
            # Sort by correlation (descending)
            high_corr_pairs = sorted(high_corr_pairs, 
                                    key=lambda x: x['correlation'], 
                                    reverse=True)
            
            self.results.append(ValidationResult(
                passed=False,
                message=f"{len(high_corr_pairs)} feature pairs with correlation >{threshold}",
                severity='warning',
                details={'high_corr_pairs': high_corr_pairs[:5]}
            ))
            print(f"[WARN] Found {len(high_corr_pairs)} feature pairs with correlation >{threshold}:")
            for pair in high_corr_pairs[:5]:
                print(f"       {pair['feature_1']} <-> {pair['feature_2']}: {pair['correlation']:.3f}")
            if len(high_corr_pairs) > 5:
                print(f"       ... and {len(high_corr_pairs) - 5} more pairs")
        else:
            self.results.append(ValidationResult(
                passed=True,
                message=f"No feature pairs with correlation >{threshold}",
                severity='info'
            ))
            print(f"[PASS] All feature correlations <{threshold}")
    
    def _check_stationarity(self):
        """Check features for stationarity using Augmented Dickey-Fuller test."""
        print("\n11. Stationarity (ADF Test)")
        print("-" * 40)
        
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
                print(f"       [WARN] Could not test {col}: {str(e)[:50]}")
                continue
        
        total_tested = stationary_count + len(non_stationary)
        
        print(f"  Features tested: {total_tested}")
        print(f"  Stationary (p < {alpha}): {stationary_count}")
        print(f"  Non-stationary (p >= {alpha}): {len(non_stationary)}")
        
        if len(non_stationary) > 0:
            # Sort by p-value (most non-stationary first)
            non_stationary = sorted(non_stationary, 
                                   key=lambda x: x['p_value'], 
                                   reverse=True)
            
            self.results.append(ValidationResult(
                passed=False,
                message=f"{len(non_stationary)} features are non-stationary (p >= {alpha})",
                severity='warning',
                details={'non_stationary': non_stationary[:5]}
            ))
            print(f"[WARN] {len(non_stationary)} non-stationary features (may have trends/unit roots):")
            for item in non_stationary[:5]:
                print(f"       {item['feature']}: p={item['p_value']:.4f}, ADF={item['adf_stat']:.3f}")
            if len(non_stationary) > 5:
                print(f"       ... and {len(non_stationary) - 5} more")
            print(f"       Note: Non-stationary features may need differencing or detrending")
        else:
            self.results.append(ValidationResult(
                passed=True,
                message=f"All features are stationary (p < {alpha})",
                severity='info'
            ))
            print(f"[PASS] All {total_tested} features are stationary")
    
    def _print_summary(self):
        """Print validation summary."""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        errors = [r for r in self.results if r.severity == 'error' and not r.passed]
        warnings = [r for r in self.results if r.severity == 'warning' and not r.passed]
        passed = [r for r in self.results if r.passed]
        
        print(f"[PASS] Passed: {len(passed)}")
        print(f"[WARN] Warnings: {len(warnings)}")
        print(f"[FAIL] Errors: {len(errors)}")
        
        if errors:
            print("\nERRORS:")
            for r in errors:
                print(f"  - {r.message}")
        
        if warnings:
            print("\nWARNINGS:")
            for r in warnings[:5]:
                print(f"  - {r.message}")
            if len(warnings) > 5:
                print(f"  - ... and {len(warnings) - 5} more warnings")
        
        print("\n" + "="*80)


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
