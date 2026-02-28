"""
Within-group PCA for macro feature selection.

Fits PCA independently per macro category (rates, inflation, growth, employment,
liquidity, stress) using IS-only data, then transforms the full dataset. The IS
boundary is enforced identically to initialise_emissions() — data is sliced to
rows <= train_end_date before fitting; OOS rows are transformed but never used
for fitting.

Output columns are named {group}_pc{n} (e.g. rates_pc1, rates_pc2, growth_pc1).
Calling get_ordered_pc_columns() returns all PC names sorted by explained variance
descending — slicing this list to [:N] gives a valid N-feature candidate set for
HMM stability selection.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class GroupPCATransformer:
    """
    Applies PCA within each macro group on IS data and transforms the full dataset.

    Args:
        n_components_per_group: Dict mapping group name to number of PCs to retain,
            e.g. {'rates': 2, 'inflation': 1, 'stress': 2}. Loaded from
            configs/regimes/regime_config.yaml feature_selection.group_pca.n_components.
        train_end_date: IS boundary (inclusive). PCA is fit only on rows with
            index <= this date. Matches the convention in initialise_emissions().
        sign_anchors: Optional dict mapping group name to anchor config, e.g.
            {'liquidity': {'series': 'VIXCLS', 'good_direction': 'low'}}. For each
            group, PC1 is sign-normalised so that positive values always indicate
            economically *good* conditions. 'good_direction' is 'low' for indicators
            where a lower value is better (VIX, UNRATE), and 'high' for indicators
            where a higher value is better (CFNAI). The sign is determined from the
            IS-fitted loading — no future data is used. The convention makes archetype
            templates written in economic intuition space (positive = good) directly
            comparable to state means.
    """

    def __init__(
        self,
        n_components_per_group: dict[str, int],
        train_end_date: str,
        sign_anchors: dict[str, dict] | None = None,
    ) -> None:
        self.n_components_per_group = n_components_per_group
        self.train_end_date = pd.Timestamp(train_end_date)
        self._sign_anchors: dict[str, dict] = sign_anchors or {}
        self._pcas: dict[str, PCA] = {}
        self._group_features: dict[str, list[str]] = {}

    def fit(
        self, features: pd.DataFrame, group_map: dict[str, str]
    ) -> "GroupPCATransformer":
        """
        Fit PCA per group using IS rows only.

        After fitting, columns produced by transform() are ordered by explained
        variance descending so that features.columns[:N] always gives the N most
        informative PCs across all groups.

        Args:
            features:  Full feature DataFrame (IS + OOS), no NaNs expected.
            group_map: {feature_name: group} e.g. {'VIXCLS_level_zscore_63': 'stress', ...}.
                       Built by build_featuregroup_map().

        Returns:
            self (for chaining with transform())
        """
        is_mask = features.index <= self.train_end_date
        n_is_rows = is_mask.sum()
        logger.info(
            "Fitting group PCA on %d IS rows (up to %s)",
            n_is_rows,
            self.train_end_date.date(),
        )

        # Gather feature names per group (preserve column order)
        groups: dict[str, list[str]] = {}
        for feat in features.columns:
            group = group_map.get(feat, "unknown")
            groups.setdefault(group, []).append(feat)

        for group, feats in sorted(groups.items()):
            n_requested = self.n_components_per_group.get(group, 1)
            n_components = min(n_requested, len(feats))

            if n_components != n_requested:
                logger.warning(
                    "Group '%s': requested %d PCs but only %d features available — capping at %d",
                    group,
                    n_requested,
                    len(feats),
                    n_components,
                )

            X_is = features.loc[is_mask, feats].values
            pca = PCA(n_components=n_components)
            pca.fit(X_is)

            # Sign-normalise PC1 so positive = economically good conditions.
            # The anchor feature's IS loading on PC1 determines the flip: if the
            # anchor is a "bad" indicator (good_direction='low') and PC1 loads
            # positively on it, high PC1 = bad → flip. Vice-versa for 'high'.
            # Only PC1 is flipped; higher PCs capture orthogonal residual variance
            # that does not have a clean economic interpretation axis.
            anchor_cfg = self._sign_anchors.get(group)
            if anchor_cfg is not None:
                anchor_series = anchor_cfg["series"]
                good_direction = anchor_cfg.get("good_direction", "high")
                # Find the first feature whose name starts with the anchor series code
                anchor_idx = next(
                    (i for i, f in enumerate(feats) if f.startswith(anchor_series)),
                    None,
                )
                if anchor_idx is not None:
                    loading = float(pca.components_[0, anchor_idx])
                    should_flip = (
                        (good_direction == "low" and loading > 0)
                        or (good_direction == "high" and loading < 0)
                    )
                    if should_flip:
                        pca.components_[0] *= -1
                        logger.info(
                            "Group '%s': flipped PC1 sign (anchor=%s, loading_before_flip=%.3f,"
                            " good_direction=%s)",
                            group,
                            anchor_series,
                            loading,
                            good_direction,
                        )
                    else:
                        logger.info(
                            "Group '%s': PC1 sign already correct (anchor=%s, loading=%.3f,"
                            " good_direction=%s)",
                            group,
                            anchor_series,
                            loading,
                            good_direction,
                        )
                else:
                    logger.warning(
                        "Group '%s': anchor series '%s' not found in features %s — skipping sign fix",
                        group,
                        anchor_series,
                        feats,
                    )

            self._pcas[group] = pca
            self._group_features[group] = feats

            var_ratios = ", ".join(
                f"PC{i+1}={r:.1%}" for i, r in enumerate(pca.explained_variance_ratio_)
            )
            logger.info(
                "Group '%s': %d features → %d PCs | explained variance: %s",
                group,
                len(feats),
                n_components,
                var_ratios,
            )

        # Pre-compute explained-variance ordering so transform() outputs columns in that order
        self._ordered_columns = self.get_ordered_pc_columns()

        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted PCA to the full dataset (IS + OOS).

        Rows with NaN values in a group's features will produce NaN PC values
        for that group (consistent with how raw features handle the burn-in period).

        Args:
            features: Full feature DataFrame. Must contain all columns used during fit().

        Returns:
            DataFrame with columns {group}_pc1, {group}_pc2, ... indexed identically
            to the input.
        """
        if not self._pcas:
            raise RuntimeError("GroupPCATransformer must be fit() before transform().")

        # Build a buffer for all PC columns, then reorder by explained variance
        buffer: dict[str, np.ndarray] = {}

        for group, pca in self._pcas.items():
            feats = self._group_features[group]
            group_data = features[feats].values  # (T, n_feats)

            # Rows where any feature in the group is NaN
            nan_mask = np.isnan(group_data).any(axis=1)

            transformed = np.full((len(features), pca.n_components_), np.nan)
            if (~nan_mask).any():
                transformed[~nan_mask] = pca.transform(group_data[~nan_mask])

            for i in range(pca.n_components_):
                buffer[f"{group}_pc{i + 1}"] = transformed[:, i]

        # Output columns in explained-variance order so features.columns[:N] == top-N PCs
        result = pd.DataFrame(index=features.index)
        for col in self._ordered_columns:
            result[col] = buffer[col]

        return result

    def fit_transform(
        self,
        features: pd.DataFrame,
        group_map: dict[str, str],
    ) -> pd.DataFrame:
        """Fit on IS data then transform full dataset. Convenience wrapper."""
        return self.fit(features, group_map).transform(features)

    def get_ordered_pc_columns(self) -> list[str]:
        """
        Return all PC column names sorted by explained variance descending.

        This is the ordered list for HMM candidate count selection:
            transformer.get_ordered_pc_columns()[:N]
        gives the N most informative PCs across all groups for a candidate
        feature count of N.
        """
        if not self._pcas:
            raise RuntimeError(
                "GroupPCATransformer must be fit() before get_ordered_pc_columns()."
            )

        all_pcs: list[tuple[str, float]] = []
        for group, pca in self._pcas.items():
            for i, var in enumerate(pca.explained_variance_):
                all_pcs.append((f"{group}_pc{i + 1}", float(var)))

        all_pcs.sort(key=lambda x: x[1], reverse=True)
        return [col for col, _ in all_pcs]

    def get_loadings(self) -> dict[str, pd.DataFrame]:
        """
        Return per-group loading matrices for interpretability.

        Returns:
            Dict mapping group name to a DataFrame with:
                - rows: original feature names
                - columns: PC1, PC2, ...
            e.g. rates loadings show how T10Y3M, DGS10, DGS2 combine into rates_pc1.
        """
        if not self._pcas:
            raise RuntimeError(
                "GroupPCATransformer must be fit() before get_loadings()."
            )

        loadings: dict[str, pd.DataFrame] = {}
        for group, pca in self._pcas.items():
            feats = self._group_features[group]
            pc_cols = [f"PC{i + 1}" for i in range(pca.n_components_)]
            # components_ shape: (n_components, n_features) → transpose to (n_features, n_components)
            loadings[group] = pd.DataFrame(
                pca.components_.T,
                index=feats,
                columns=pc_cols,
            )
        return loadings

    def save_loadings(self, directory: Path) -> None:
        """
        Persist PCA loadings and explained variance to CSV for post-run inspection.

        Writes:
            {directory}/pca_loadings_{group}.csv   — one file per group
            {directory}/pca_explained_variance.csv  — global summary across all groups

        Args:
            directory: Target directory (created by caller before this call).
        """
        loadings = self.get_loadings()

        for group, loading_df in loadings.items():
            path = directory / f"pca_loadings_{group}.csv"
            loading_df.to_csv(path)
            logger.debug("Saved %s loadings to %s", group, path)

        # Global explained variance summary
        rows = []
        for group, pca in self._pcas.items():
            for i, (var, ratio) in enumerate(
                zip(pca.explained_variance_, pca.explained_variance_ratio_)
            ):
                rows.append(
                    {
                        "group": group,
                        "pc": f"PC{i + 1}",
                        "column": f"{group}_pc{i + 1}",
                        "explained_variance": round(float(var), 6),
                        "explained_variance_ratio": round(float(ratio), 4),
                    }
                )

        summary_df = pd.DataFrame(rows).sort_values(
            "explained_variance", ascending=False
        )
        summary_path = directory / "pca_explained_variance.csv"
        summary_df.to_csv(summary_path, index=False)

        logger.info(
            "Saved PCA loadings for %d groups + explained variance summary to %s",
            len(loadings),
            directory,
        )
