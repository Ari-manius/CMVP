"""
CMVP Backbone Validation - Null Calibration & Degree-Bias Diagnostics
========================================================================

_NullCalibrationMixin, mixed into BackboneValidator (see backbone_validation.py).
Plots and helpers that compare observed statistics against the exact/sampled
null distribution and check for residual degree bias after UBCM correction.
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from typing import Dict, Optional

from .validation import (PlotConfig, _finalize_plot, _configure_scatter_plot,
                         sample_adjacency_from_p, get_categorical_color_map)


class _NullCalibrationMixin:
    """Null-model calibration and degree-bias diagnostic plots for BackboneValidator."""

    def plot_empirical_vs_null(self, node_pairs: Optional[list] = None,
                              n_samples: int = 100, save_path: Optional[str] = None,
                              show: bool = True):
        """
        Plot empirical vs null distribution using NEMtropy sampler.

        Generates random graphs from the null model using NEMtropy's sampler
        and compares observed similarity to null distribution.

        Parameters
        ----------
        node_pairs : list of tuples, optional
            Node index pairs (i, j) to visualize. If None, auto-selects 3 pairs
            with low/median/high p-values
        n_samples : int
            Number of random graphs to sample from null model (default: 100)
        save_path : str, optional
            Path to save the figure
        show : bool
            Whether to display the plot
        """
        # Use observed similarity from core.py
        if self.obs_sim is None:
            # Fallback: compute common neighbors if not cached
            obs_cn = (self.A > 0).astype(int)
            obs_cn = obs_cn @ obs_cn
            np.fill_diagonal(obs_cn, 0)
        else:
            obs_cn = self.obs_sim

        # Select representative node pairs
        if node_pairs is None:
            # Get candidate pairs (non-zero expected)
            if self.directed:
                mask = ~np.eye(self.n, dtype=bool)
            else:
                mask = np.triu(np.ones((self.n, self.n), dtype=bool), k=1)

            all_indices = np.where(mask)
            pvals_all = np.asarray(self.pvalues[mask]).ravel()

            # Drop NaN pairs (untested)
            valid = np.isfinite(pvals_all)
            pvals_candidates = pvals_all[valid]
            indices = (all_indices[0][valid], all_indices[1][valid])

            # Select 3 representative pairs: lowest, closest to midpoint, highest p-value
            if len(pvals_candidates) >= 3:
                sorted_idx = np.argsort(pvals_candidates)
                sorted_pvals = pvals_candidates[sorted_idx]
                mid_target = (sorted_pvals[0] + sorted_pvals[-1]) / 2
                mid_pos = np.argmin(np.abs(sorted_pvals - mid_target))
                # Avoid exact p=0 for low — pick smallest nonzero if available
                nonzero = sorted_idx[sorted_pvals[np.arange(len(sorted_idx))] > 0]
                low_idx = nonzero[0] if len(nonzero) > 0 else sorted_idx[0]
                indices_list = [
                    low_idx,              # Smallest nonzero p-value
                    sorted_idx[mid_pos],  # Closest to midpoint of p-value range
                    sorted_idx[-1]        # Least significant
                ]
                node_pairs = [(indices[0][k], indices[1][k]) for k in indices_list]
                node_list = list(self.G.nodes())
                print("Auto-selected node pairs by p-value:")
                for k, idx in enumerate(indices_list):
                    i, j = indices[0][idx], indices[1][idx]
                    print(f"  {k+1}. ({node_list[i]}, {node_list[j]}): p={pvals_candidates[idx]:.4f}")
                print()
            else:
                node_pairs = list(zip(indices[0][:3], indices[1][:3]))

        n_pairs = len(node_pairs)
        fig, axes = plt.subplots(1, n_pairs, figsize=(6*n_pairs, 5))
        if n_pairs == 1:
            axes = [axes]

        node_list = list(self.G.nodes())

        # Sample null distributions using NEMtropy
        null_cn_samples = self._sample_null_cn(node_pairs, n_samples)

        for idx, (i, j) in enumerate(node_pairs):
            ax = axes[idx]
            node_i, node_j = node_list[i], node_list[j]

            # Get observed value
            obs_val = obs_cn[i, j]

            # Get expected value — use stored analytical result if available
            exp_matrix = self.cmvp.exp_sim_matrix
            exp_val = exp_matrix[i, j] if exp_matrix is not None else self.cmvp.p_matrix[i, :] @ self.cmvp.p_matrix[j, :]

            # Get null samples for this pair
            null_samples = null_cn_samples[idx]
            # Plot histogram — discrete integer bins, bars touching
            int_samples = null_samples.astype(int)
            lo, hi = int_samples.min(), int_samples.max()
            bins = np.arange(lo, hi + 2) - 0.5  # half-integer edges so each bar = one count
            counts, _ = np.histogram(int_samples, bins=bins)
            bin_vals = np.arange(lo, hi + 1)

            BAR_COLOR = PlotConfig.ACCENT      # sampling / bars
            ANA_COLOR = PlotConfig.PRIMARY     # analytical

            ax.bar(bin_vals, counts, width=1.0, alpha=0.8,
                   color=BAR_COLOR, edgecolor='white', linewidth=0.3)

            x_lo = min(lo, int(obs_val)) - 1
            x_hi = max(hi, int(obs_val)) + 1
            step = max(1, (x_hi - x_lo) // 10)
            ax.set_xticks(np.arange(x_lo, x_hi + step, step))
            ax.set_xlim(x_lo - 0.5, x_hi + 0.5)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))

            # Sampling statistics
            null_mean = np.mean(null_samples)
            null_std = np.std(null_samples)
            null_se = null_std / np.sqrt(len(null_samples))

            # Analytical statistics
            exp_std_val = self.cmvp.exp_sim_matrix_std[i, j] \
                          if self.cmvp.exp_sim_matrix_std is not None else None

            y_max = counts.max()

            # Observed
            ax.axvline(obs_val, color='red', linestyle='-', linewidth=2.5,
                      label=f'Observed = {obs_val:.0f}', zorder=5)

            # Analytical: mean + ±1 std shaded band
            ax.axvline(exp_val, color=ANA_COLOR, linestyle='--', linewidth=2,
                      label=f'Analytical μ = {exp_val:.2f}', zorder=4)
            if exp_std_val is not None:
                ax.axvspan(exp_val - exp_std_val, exp_val + exp_std_val,
                          alpha=0.15, color=ANA_COLOR,
                          label=f'Analytical ±1σ = {exp_std_val:.2f}')

            # Sampling: mean + ±1 std shaded band (same color as bars)
            ax.axvline(null_mean, color=BAR_COLOR, linestyle=':', linewidth=2,
                      label=f'Sampling μ = {null_mean:.2f} (SE={null_se:.2f})', zorder=4)
            ax.axvspan(null_mean - null_std, null_mean + null_std,
                      alpha=0.15, color=BAR_COLOR,
                      label=f'Sampling ±1σ = {null_std:.2f}')

            p_val = float(self.pvalues[i, j])
            ax.text(0.02, 0.97, f'p = {p_val:.4f}', transform=ax.transAxes,
                   fontsize=PlotConfig.FONT_SIZE_ANNOTATION, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            ax.set_xlabel('Common Neighbors', fontweight='bold')
            ax.set_ylabel('Count', fontweight='bold')
            ax.set_title(f'({node_i}, {node_j})', fontweight='bold')
            ax.legend(fontsize=PlotConfig.FONT_SIZE_TICK)
            ax.grid(alpha=PlotConfig.GRID_ALPHA)

        plt.suptitle(f'Null Distribution from NEMtropy Sampler (n={n_samples})',
                    **PlotConfig.SUPTITLE_KW)

        _finalize_plot(fig, save_path, show)

    def _sample_null_cn(self, node_pairs: list, n_samples: int) -> Dict:
        """
        Sample common neighbors from null model using NEMtropy.

        Uses the fitted null model to generate random graphs and compute
        common neighbors for specified node pairs.

        Parameters
        ----------
        node_pairs : list
            List of (i, j) node index pairs
        n_samples : int
            Number of samples to draw

        Returns
        -------
        dict
            Dictionary with sampled common neighbors for each pair
        """
        null_cn_dict = {idx: [] for idx in range(len(node_pairs))}
        rng = np.random.default_rng(self.cmvp.seed)

        # NEMtropy's fitted graph objects don't expose a direct .sample()
        # method, so this always falls through to Bernoulli sampling from
        # p_matrix (kept as p_matrix is the analytically exact null anyway).
        use_null_model_sampler = (
            hasattr(self.cmvp, 'null_model') and self.cmvp.null_model is not None
            and hasattr(self.cmvp.null_model, 'sample')
        )

        for sample_idx in range(n_samples):
            if use_null_model_sampler:
                try:
                    adj = self.cmvp.null_model.sample()
                except Exception:
                    adj = sample_adjacency_from_p(self.cmvp.p_matrix, self.directed, rng)
            else:
                adj = sample_adjacency_from_p(self.cmvp.p_matrix, self.directed, rng)

            # Compute common neighbors
            cn = (adj > 0).astype(int) @ (adj > 0).astype(int)
            np.fill_diagonal(cn, 0)

            # Extract values for each pair
            for pair_idx, (i, j) in enumerate(node_pairs):
                null_cn_dict[pair_idx].append(cn[i, j])

        # Convert to numpy arrays
        return {idx: np.array(vals) for idx, vals in null_cn_dict.items()}

    def plot_shared_neighbor_structure(self, stat: str = 'zscore', color_by: str = 'exp_cn',
                                       highlight_inter_block: bool = True,
                                       save_path: Optional[str] = None, show: bool = True):
        """
        Scatter: x = Jaccard similarity / resource allocation, y = z-score or
        effect size. Each point is one pair with CN > 0.

        Parameters
        ----------
        stat : str
            'zscore' (default), 'effect', or 'cn' — which statistic to put on
            the y-axis (see _stat_matrix()).
        color_by : str
            'exp_cn' (default) — E[CN] under the null model;
            'std'              — std[CN] under the null model;
            'cn'               — observed CN;
            anything else      — treated as a node attribute name (e.g.
                                  'block' for an SBM ground truth), coloring
                                  each pair categorically by its shared block
                                  membership. Requires the attribute to be set
                                  on every node in `self.G`.
        highlight_inter_block : bool
            Only used when `color_by` names a node attribute. If True
            (default), pairs whose two endpoints sit in different categories
            are drawn as hollow markers in a fixed highlight color instead of
            being colored by (an arbitrary one of) their two categories —
            makes cross-block ties visually distinct from intra-block ones.
        """
        deg = np.asarray(self.cmvp.A_bin.sum(axis=1)).flatten()
        n = self.cmvp.n
        A = self.cmvp.A_bin.astype(float)
        A_dense = np.asarray(A.todense()) if hasattr(A, 'todense') else A
        node_list = list(self.G.nodes())

        categorical = color_by not in ('exp_cn', 'std', 'cn')
        if categorical:
            attr_values = nx.get_node_attributes(self.G, color_by)
            if len(attr_values) != n:
                raise ValueError(
                    f"node attribute '{color_by}' is missing on "
                    f"{n - len(attr_values)} of {n} nodes; set it on every node "
                    "before using it as color_by."
                )
            attr_arr = np.array([attr_values[node] for node in node_list], dtype=object)

        if self.directed:
            ii, jj = np.where(~np.eye(n, dtype=bool))
        else:
            ii, jj = np.triu_indices(n, k=1)

        deg_product = deg[ii] * deg[jj]
        sum_deg_vals = np.full(len(ii), np.nan)
        sum_inv_deg_vals = np.full(len(ii), np.nan)
        adamic_adar_vals = np.full(len(ii), np.nan)
        resource_alloc_vals = np.full(len(ii), np.nan)

        for k, (i, j) in enumerate(zip(ii, jj)):
            shared = np.where((A_dense[i] > 0) & (A_dense[j] > 0))[0]
            if len(shared) == 0:
                continue
            d = deg[shared].astype(float)
            sum_deg_vals[k] = d.sum()
            sum_inv_deg_vals[k] = (d ** 2).mean()
            with np.errstate(divide='ignore', invalid='ignore'):
                adamic_adar_vals[k] = np.sum(1.0 / np.log(np.where(d > 1, d, np.nan)))
                resource_alloc_vals[k] = np.sum(1.0 / np.where(d > 0, d, np.nan))

        # Per-pair z-score or effect size
        obs_mat = self.cmvp.obs_sim_matrix
        exp_mat = self.cmvp.exp_sim_matrix
        std_mat = self.cmvp.exp_sim_matrix_std
        stat_matrix, stat_label = self._stat_matrix(stat)
        np.fill_diagonal(stat_matrix, np.nan)
        z_vals = stat_matrix[ii, jj]

        # Jaccard per pair: CN / (deg_i + deg_j - CN)
        cn_flat = obs_mat[ii, jj] if obs_mat is not None else np.zeros(len(ii))
        union = deg[ii] + deg[jj] - cn_flat
        with np.errstate(invalid='ignore', divide='ignore'):
            jaccard_vals = np.where(union > 0, cn_flat / union, np.nan)

        exp_cn_vals = exp_mat[ii, jj] if exp_mat is not None else np.zeros(len(ii))
        std_vals_all = std_mat[ii, jj] if std_mat is not None else np.zeros(len(ii))
        valid = ~np.isnan(resource_alloc_vals) & ~np.isnan(jaccard_vals) & ~np.isnan(z_vals) & ~np.isnan(exp_cn_vals)

        j = jaccard_vals[valid]
        ra = resource_alloc_vals[valid]

        z = z_vals[valid]
        ecn = exp_cn_vals[valid]
        std_vals = std_vals_all[valid]
        cn_vals = cn_flat[valid]

        scatter_kw = dict(s=20, alpha=PlotConfig.SCATTER_ALPHA, linewidth=0, rasterized=True)

        if categorical:
            pair_attr_i = attr_arr[ii][valid]
            pair_attr_j = attr_arr[jj][valid]
            same_block = pair_attr_i == pair_attr_j
            block_color_map = get_categorical_color_map(attr_arr)

            fig, axes = plt.subplots(1, 2, figsize=PlotConfig.WIDE)
            axes[1].sharey(axes[0])

            inter_color = 'crimson'
            for ax, x_vals in zip(axes, (j, ra)):
                for block_val, color in block_color_map.items():
                    mask = same_block & (pair_attr_i == block_val)
                    if mask.any():
                        ax.scatter(x_vals[mask], z[mask], color=color,
                                  label=f'{color_by}={block_val}', **scatter_kw)
                if highlight_inter_block and (~same_block).any():
                    ax.scatter(x_vals[~same_block], z[~same_block], facecolors='none',
                              edgecolors=inter_color, linewidth=0.8, s=24,
                              alpha=PlotConfig.SCATTER_ALPHA, rasterized=True,
                              label='inter-block', zorder=5)
                elif (~same_block).any():
                    ax.scatter(x_vals[~same_block], z[~same_block], color='lightgray',
                              label='inter-block', **scatter_kw)

            axes[0].set_xlabel('Jaccard similarity')
            axes[1].set_xlabel('Resource allocation  (Σ 1/deg(k))')
            axes[1].legend(fontsize=PlotConfig.FONT_SIZE_TICK, loc='upper right')
            color_label = f'{color_by} (same-block; hollow = inter-block)' \
                          if highlight_inter_block else color_by
        else:
            color_vals, color_label = self._color_by_std_cn_expcn(color_by, std_vals, cn_vals, ecn)

            # Dedicate a narrow 3rd GridSpec column to the colorbar so it's a real subplot
            # axis (not a hand-floated one) -- keeps it outside the scatter panels even
            # after _finalize_plot()'s tight_layout() call.
            fig, axes = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1, 0.05]})
            axes, cax = axes[:2], axes[2]
            axes[1].sharey(axes[0])

            axes[0].scatter(j, z, c=color_vals, cmap='viridis', **scatter_kw)
            axes[0].set_xlabel('Jaccard similarity')

            sc = axes[1].scatter(ra, z, c=color_vals, cmap='viridis', **scatter_kw)
            axes[1].set_xlabel('Resource allocation  (Σ 1/deg(k))')

            cbar = fig.colorbar(sc, cax=cax)
            cbar.set_label(color_label, rotation=270, labelpad=15)

        for ax, x_vals in zip(axes, (j, ra)):
            ax.grid(alpha=PlotConfig.GRID_ALPHA)
            pear_r, _ = pearsonr(x_vals, z)
            spear_r, _ = spearmanr(x_vals, z)
            ax.text(0.02, 0.98, f'Pearson r = {pear_r:.3f}\nSpearman ρ = {spear_r:.3f}',
                    transform=ax.transAxes, ha='left', va='top', fontsize=PlotConfig.FONT_SIZE_ANNOTATION,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        axes[0].set_ylabel(stat_label)

        fig.suptitle('CMVP vs. Similarity Measures', **PlotConfig.SUPTITLE_KW)
        _finalize_plot(fig, save_path, show)

    def _x_for_degree_interpolator(self):
        """
        Build a degree -> fitted UBCM fitness x interpolator (log-linear in
        degree, extrapolated beyond the observed range using the nearest
        boundary's slope). Shared by plot_jaccard_null_cdf and
        plot_zmax_ceiling_vs_degree for constructing synthetic same-degree
        pairs. Requires a fitted undirected unweighted null model exposing
        cmvp.x.

        Returns
        -------
        x_for_degree : callable
            d -> fitted fitness x at degree d.
        uniq_degs : ndarray
            The graph's observed unique positive degrees (the interpolator's
            actual support — values outside [uniq_degs[0], uniq_degs[-1]]
            are extrapolated, not interpolated).
        """
        has_x = hasattr(self.cmvp, 'x') and self.cmvp.x is not None
        if not has_x:
            raise ValueError(
                "requires a fitted undirected unweighted null model (model='cm' or "
                "'cm_exp') exposing fitted fitness parameters on cmvp.x."
            )
        deg = np.asarray(self.cmvp.A_bin.sum(axis=1)).flatten()
        x_real = np.asarray(self.cmvp.x, dtype=float)

        uniq_degs, inverse = np.unique(deg, return_inverse=True)
        log_x_by_deg = np.array([
            np.log(x_real[inverse == idx]).mean() for idx in range(len(uniq_degs))
        ])

        def x_for_degree(d):
            if d <= uniq_degs[0]:
                lo, hi = 0, min(1, len(uniq_degs) - 1)
            elif d >= uniq_degs[-1]:
                lo, hi = max(0, len(uniq_degs) - 2), len(uniq_degs) - 1
            else:
                return np.exp(np.interp(d, uniq_degs, log_x_by_deg))
            if lo == hi:
                return np.exp(log_x_by_deg[lo])
            slope = (log_x_by_deg[hi] - log_x_by_deg[lo]) / (uniq_degs[hi] - uniq_degs[lo])
            anchor = uniq_degs[hi] if d >= uniq_degs[-1] else uniq_degs[lo]
            anchor_log_x = log_x_by_deg[hi] if d >= uniq_degs[-1] else log_x_by_deg[lo]
            return np.exp(anchor_log_x + slope * (d - anchor))

        return x_for_degree, uniq_degs

    def plot_zmax_ceiling_vs_degree(self, degrees=None, n_degrees: int = 60,
                                    deg_min: Optional[float] = None, deg_max: Optional[float] = None,
                                    stat: str = 'zscore',
                                    save_path: Optional[str] = None, show: bool = True):
        """
        Ceiling and floor significance/effect size for synthetic equal-degree
        pairs swept across a range of degree — the endpoints of each pair's
        discrete support:

        z_max(k) = (k - E[w]) / std[w]  at w = w_max = k (Jaccard = 1,
        fully overlapping neighborhoods).
        z_min(k) = (0 - E[w]) / std[w]  at w = 0 (Jaccard = 0, no shared
        neighbors at all). (/Var[w] instead of /std[w] for stat='effect'.)

        E[w] = sum(p_ik) and Var[w] = sum(p_ik*(1-p_ik)) are exact closed
        forms (linearity of expectation / independence under the null), so
        this needs no pmf convolution — cheap enough to sweep many degree
        values.

        In the sparse regime (low degree, small p_ik), Var[w] ~ E[w]
        (Poisson-like), so z_max grows roughly like sqrt(k) as degree
        increases, while z_min ~ -sqrt(E[w]) shrinks toward 0 (there's
        little room below an already-small mean). As degree grows further
        and p_ik stops being small, p(1-p) is maximized near p=0.5 rather
        than continuing to track p, so Var[w] grows sub-linearly relative
        to the sparse regime while w_max=k keeps growing linearly — z_max
        can then peak at some intermediate degree and decline again at
        higher degree, rather than growing monotonically, while z_min keeps
        growing in magnitude (E[w] itself grows with degree, giving more
        room below it). That non-monotonicity is a real statement about the
        null model's discriminating power, not a plotting artifact — see
        plot_jaccard_null_cdf's right-hand panel for the z-score curves
        these quantities bound.

        Parameters
        ----------
        degrees : array-like, optional
            Degree values to sweep. Defaults to `n_degrees` points
            log-spaced between `deg_min` and `deg_max`.
        n_degrees : int
            Number of points when `degrees` is None.
        deg_min, deg_max : float, optional
            Lower/upper end of the default log-spaced sweep. Each defaults
            to the graph's observed min/max degree (`uniq_degs[0]`/`[-1]`)
            when not given — pass either to sweep below/above the observed
            range. Degrees outside the observed range use the
            interpolator's extrapolation (see `_x_for_degree_interpolator`);
            the observed range is still marked (shaded/dotted lines) on the
            plot regardless of these bounds. Ignored if `degrees` is given
            explicitly.
        stat : {'zscore', 'effect', 'both'}
            'zscore': divide by std[w]. 'effect': divide by Var[w]. 'both':
            plot zscore (solid) and effect (dashed) together on one shared
            y-axis — note the two are on different scales (std- vs
            Var-normalized), so overlaying them this way is only a
            qualitative comparison of shape, not magnitude.
        """
        if stat not in ('zscore', 'effect', 'both'):
            raise ValueError("stat must be 'zscore', 'effect' or 'both'")
        x_for_degree, uniq_degs = self._x_for_degree_interpolator()
        x_real = np.asarray(self.cmvp.x, dtype=float)

        if degrees is None:
            # Degree is integer-valued, so log-spacing then rounding (and
            # de-duplicating) avoids requesting fractional "degrees" that no
            # real node could have.
            lo = deg_min if deg_min is not None else uniq_degs[0]
            hi = deg_max if deg_max is not None else uniq_degs[-1]
            degrees = np.unique(np.round(np.geomspace(max(lo, 1.0), hi, n_degrees)))
        degrees = np.asarray(sorted(degrees), dtype=float)

        stats = ('zscore', 'effect') if stat == 'both' else (stat,)
        vals = {s: (np.full(len(degrees), np.nan), np.full(len(degrees), np.nan))
                for s in stats}
        for idx, d in enumerate(degrees):
            x_s = x_for_degree(d)
            p_row = x_s * x_real / (1.0 + x_s * x_real)
            pk = np.asarray(p_row * p_row).flatten()
            pk = pk[(pk > 0) & (pk < 1)]

            mean_v = np.sum(pk)
            var_v = np.sum(pk * (1.0 - pk))
            for s in stats:
                denom = np.sqrt(var_v) if s == 'zscore' else var_v
                if denom > 0:
                    vals[s][0][idx] = (d - mean_v) / denom
                    vals[s][1][idx] = (0.0 - mean_v) / denom

        fig, ax = plt.subplots(figsize=PlotConfig.SMALL)

        style = {'zscore': dict(max=('mediumpurple', '-'), min=('mediumpurple', '-')),
                 'effect': dict(max=('teal', '-'), min=('teal', '-'))}
        stat_label = {'zscore': 'z', 'effect': r'\widehat{\mathrm{Effect}}'}
        for s in stats:
            z_max_vals, z_min_vals = vals[s]
            c_max, ls_max = style[s]['max']
            c_min, ls_min = style[s]['min']
            suffix = f' ({s})' if stat == 'both' else ''
            ax.plot(degrees, z_max_vals, color=c_max, lw=2, ls=ls_max, marker='o', markersize=3,
                    label=f'${stat_label[s]}_{{max}}$ (w=k, J=1){suffix}')
            ax.plot(degrees, z_min_vals, color=c_min, lw=2, ls=ls_min, marker='o', markersize=3,
                    label=f'${stat_label[s]}_{{min}}$ (w=0, J=0){suffix}')

        ax.axhline(0, color='k', lw=0.8, ls=':', alpha=0.4)
        ax.axvspan(degrees.min(), uniq_degs[0], color='grey', alpha=0.08)
        ax.axvspan(uniq_degs[-1], degrees.max(), color='grey', alpha=0.08)
        ax.axvline(uniq_degs[0], color='k', lw=0.8, ls=':', alpha=0.4)
        ax.axvline(uniq_degs[-1], color='k', lw=0.8, ls=':', alpha=0.4,
                   label='observed degree range (shaded = extrapolated)')
        ax.set_xlabel('Degree $k$ (equal-degree synthetic pair)')
        if stat == 'both':
            ax.set_ylabel('z-score / Effect size (at w=k and w=0)')
        else:
            ax.set_ylabel(f'{"z-score" if stat == "zscore" else "Effect size"} (at w=k and w=0)')
        title = 'Null Model Ceiling/Floor'
        ax.set_title(title)
        ax.grid(alpha=PlotConfig.GRID_ALPHA)
        ax.legend(fontsize=PlotConfig.FONT_SIZE_TICK)
        _finalize_plot(fig, save_path, show)

    def plot_jaccard_null_cdf(self, pairs=None, n_pairs: int = 6, equal_degree: bool = False,
                              degrees=None, stat: str = 'zscore', normalize_jaccard: bool = True,
                              tail_prob: float = 0.001, tail_std: Optional[float] = None,
                              save_path: Optional[str] = None, show: bool = True):
        """
        Exact null CDF of Jaccard similarity J_ik for a handful of pairs,
        spanning the E[CN] range.

        J_ik = w_ik / (k_i + k_k - w_ik) is monotone increasing in w_ik at
        fixed degrees (k_i, k_k), so the exact Poisson-Binomial pmf of w_ik
        under the null (UBCM) is pushed through that map directly — no
        simulation. Reproduces the "same J, different meaning" effect: the
        0.5-crossing of the null CDF shifts with degree, so an identical
        observed J means something different for a high- vs low-degree pair.

        Synthetic (equal_degree/degrees) pairs are two identical UBCM twins
        at the target degree, reconstructed from a fitness x_s interpolated/
        extrapolated from the real fitted x's (see
        `_x_for_degree_interpolator`). Since both twins share x_s, their
        probability of both connecting to any given real node j is
        pk_j = p_ij * p_kj = (fitted edge prob to j)^2 — one independent
        Bernoulli trial per candidate third-party node j, so w (common
        neighbors) is Poisson-Binomial: w = sum_j B_j, B_j ~ Bernoulli(pk_j).
        Its exact pmf is the convolution of every individual Bernoulli pmf
        (`pmf = convolve(pmf, [1-p, p])` per j) — not a normal or Poisson
        approximation. E[w] = sum_j pk_j and Var[w] = sum_j pk_j*(1-pk_j)
        follow directly from that (linearity of expectation / independence),
        with no need to touch the convolved pmf itself for the first two
        moments. In words: E[w] isn't an average of anything — it's a sum,
        over every other node j in the network, of how likely j is to end up
        a *shared* neighbor of the pair; nodes where both endpoints connect
        with high probability contribute close to 1 each, nodes where either
        is unlikely contribute close to 0, and adding all those
        contributions up gives the expected common-neighbor count.

        Parameters
        ----------
        pairs : list of (int, int), optional
            Node index pairs to plot. If None, `n_pairs` pairs are chosen
            automatically, evenly spaced by E[CN] across all pairs with
            E[CN] > 0.
        n_pairs : int
            Number of auto-selected pairs when `pairs` is None.
        equal_degree : bool
            If True (and `pairs` is None), ignore real node pairs entirely
            and instead build `n_pairs` *artificial* same-degree pairs at
            degree values spaced across quantiles of the graph's actual
            degree distribution (excluding the very lowest quantile). Each
            pair is two identical synthetic twins reconstructed from the
            fitted UBCM fitness parameters (see `degrees`). This isolates
            the pure degree effect on the null distribution, without any
            particular real pair's idiosyncratic third-node overlap.
        degrees : list of float, optional
            Explicit degree values for artificial same-degree pairs
            (implies `equal_degree`). Values do NOT need to lie within the
            graph's observed degree range — a fitness x_i is reconstructed
            for each target degree by interpolating/extrapolating log(x)
            against degree from the fitted null model's real nodes (linear
            in log-x, extrapolated using the nearest boundary slope beyond
            the observed range). Requires an undirected unweighted null
            model (`cm`/`cm_exp`) with fitted `cmvp.x`.
        stat : {'zscore', 'effect'}
            CMVP statistic shown in the middle/right panels in place of
            raw common-neighbor counts. 'zscore' = (w - E[w]) / std[w]
            (CMVP's test statistic). 'effect' = (w - E[w]) / Var[w]
            (CMVP's effect size, `test_sim_matrix_effect`) — same
            numerator, but scaled by variance instead of standard
            deviation, so it shrinks faster for high-degree/high-variance
            pairs than the z-score does.
        normalize_jaccard : bool
            If True (default), each line's Jaccard values are divided by
            that pair's own attainable maximum J_max = min(k_i,k_k) /
            max(k_i,k_k), so every line spans the same [0, 1] x-range
            regardless of degree. If False, raw (un-normalized) Jaccard
            values are plotted instead, on a single shared scale across
            lines — x=1.0 then means the same absolute J for every pair,
            but high-degree pairs may never reach it.
        tail_prob : float
            Tail probability trimmed from each end of every pair's null
            distribution before plotting/axis-scaling (default 0.001,
            i.e. each line/axis shows the 0.1%-99.9% probability-mass
            window of v). Lower this (e.g. 0.0001) to extend how far the
            CMVP-vs-Jaccard lines and axes reach into the tails; raise it
            to trim further in. Gets unstable/jumpy below ~1e-6: deep in a
            discrete Poisson-Binomial tail, P(v=k+1)/P(v=k) shrinks
            multiplicatively, so consecutive tail probabilities can differ
            by many orders of magnitude — a small change in tail_prob can
            then land in the same v bucket or skip several, with no smooth
            in-between. Use `tail_std` instead for a stable, continuously
            adjustable window that far out.
        tail_std : float, optional
            If given, overrides tail_prob and windows each pair's null
            distribution by |z| <= tail_std (standard deviations from the
            mean) instead of by probability mass. Since this is linear in
            the quantity being bounded, it stays smooth arbitrarily far
            into the tail — unlike tail_prob, which becomes unstable there.
        """
        if stat not in ('zscore', 'effect'):
            raise ValueError("stat must be 'zscore' or 'effect'")
        P = self.cmvp.p_matrix
        deg = np.asarray(self.cmvp.A_bin.sum(axis=1)).flatten()
        n = self.cmvp.n
        exp_mat = self.cmvp.exp_sim_matrix
        synthetic = False
        has_x = hasattr(self.cmvp, 'x') and self.cmvp.x is not None
        x_real = np.asarray(self.cmvp.x, dtype=float) if has_x else None

        if pairs is None and (equal_degree or degrees is not None):
            synthetic = True
            x_for_degree, uniq_degs = self._x_for_degree_interpolator()

            if degrees is not None:
                target_degs = np.asarray(degrees, dtype=float)
            else:
                deg_pos = deg[deg > 0]
                target_degs = np.quantile(deg_pos, np.linspace(0, 1, n_pairs + 1)[1:])

            synth_specs = []
            for d in sorted(target_degs):
                x_s = x_for_degree(d)
                p_row = x_s * x_real / (1.0 + x_s * x_real)
                synth_specs.append((d, x_s, p_row))
            pairs = synth_specs
        elif pairs is None:
            if self.directed:
                ii, jj = np.where(~np.eye(n, dtype=bool))
            else:
                ii, jj = np.triu_indices(n, k=1)
            exp_v = exp_mat[ii, jj]
            valid = exp_v > 0
            ii, jj, exp_v = ii[valid], jj[valid], exp_v[valid]
            order = np.argsort(exp_v)
            quantile_idx = np.linspace(0, len(order) - 1, n_pairs).round().astype(int)
            chosen = order[quantile_idx]
            pairs = list(zip(ii[chosen].tolist(), jj[chosen].tolist()))

        fig, (ax, ax_cn, ax_jz) = plt.subplots(1, 3, figsize=PlotConfig.THREE_PANEL)

        if has_x:
            fig2, (ax_fit, ax_pmf) = plt.subplots(1, 2, figsize=PlotConfig.WIDE)
            deg_pos_mask = deg > 0
            ax_fit.scatter(deg[deg_pos_mask], x_real[deg_pos_mask], color='lightgray',
                           s=15, alpha=0.6, zorder=1, label='real nodes')
        else:
            fig2, (ax_pmf,) = plt.subplots(1, 1, figsize=PlotConfig.SMALL)
            ax_fit = None
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(pairs)))
        z_abs_max = 0.0
        j_axis_max = 0.0

        for pair_spec, color in zip(pairs, colors):
            if synthetic:
                d, x_s, p_row = pair_spec
                pk = np.asarray(p_row * p_row).flatten()
                deg_i = deg_k = d
                if ax_fit is not None:
                    ax_fit.scatter([d], [x_s], color=color, s=50, zorder=5, edgecolor='k', linewidth=0.5)
            else:
                i, k = pair_spec
                pk = np.asarray(P[i, :] * P[k, :]).flatten()
                deg_i, deg_k = deg[i], deg[k]
                if ax_fit is not None:
                    ax_fit.scatter([deg_i, deg_k], [x_real[i], x_real[k]], color=color, s=50,
                                   zorder=5, edgecolor='k', linewidth=0.5)
            pk = pk[(pk > 0) & (pk < 1)]

            pmf = np.array([1.0])
            for p in pk:
                pmf = np.convolve(pmf, [1.0 - p, p])
            v_vals = np.arange(len(pmf))
            cdf = np.cumsum(pmf)

            mean_v = np.sum(pmf * v_vals)
            var_v = np.sum(pk * (1.0 - pk))
            std_v = np.sqrt(var_v)
            stat_denom = std_v if stat == 'zscore' else var_v

            # Probability-mass window actually worth showing — the
            # soft-constraint null lets v range further than min(k_i, k_k)
            # with vanishing probability, which would otherwise blow up
            # J = v/(k_i+k_k-v) as the union denominator nears zero. Used to
            # bound axes without distorting the CDFs.
            if tail_std is not None and std_v > 0:
                # Windowing directly in std units stays smooth arbitrarily
                # far into the tail, unlike tail_prob (see docstring) — a
                # discrete Poisson-Binomial's tail probabilities can jump by
                # orders of magnitude between adjacent v, so tiny tail_prob
                # changes there can skip several v steps or none at all.
                lo = max(int(np.floor(mean_v - tail_std * std_v)), 0)
                hi = min(int(np.ceil(mean_v + tail_std * std_v)), len(v_vals) - 1)
            else:
                lo = np.searchsorted(cdf, tail_prob)
                hi = min(np.searchsorted(cdf, 1.0 - tail_prob), len(v_vals) - 1)

            union = deg_i + deg_k - v_vals
            with np.errstate(invalid='ignore', divide='ignore'):
                j_vals = np.where(union > 0, v_vals / union, np.nan)

            # Max attainable J for this pair (at v = min(k_i, k_k)).
            v_max = min(deg_i, deg_k)
            j_max = v_max / (deg_i + deg_k - v_max) if (deg_i + deg_k - v_max) > 0 else np.nan
            if normalize_jaccard:
                # Divide by each pair's own ceiling so curves are comparable
                # in [0, 1] regardless of the degree-imposed ceiling.
                j_vals = j_vals / j_max
                j_axis_max = 1.0
            else:
                # Shared, un-normalized Jaccard scale across all lines,
                # bounded to the visible probability-mass window.
                j_axis_max = max(j_axis_max, np.nanmax(j_vals[lo:hi + 1]))

            cross_idx = np.searchsorted(cdf, 0.5)
            cross_idx = min(cross_idx, len(j_vals) - 1)

            if synthetic:
                label = f'$k_{{i,j}}$={deg_i:.1f}  $E[CN_{{ij}}]$={mean_v:.1f}'
                label_main = f'$k_{{i,j}}$={deg_i:.1f}'
            else:
                label = f'({i},{k}): $k_{{i,j}}$={int(deg_i)},{int(deg_k)}  $E[CN_{{ij}}]$={exp_mat[i, k]:.1f}'
                label_main = f'({i},{k}): $k_{{i,j}}$={int(deg_i)},{int(deg_k)}'
            ax.step(j_vals, cdf, where='post', color=color, label=label_main)
            ax.scatter([j_vals[cross_idx]], [0.5], color=color, zorder=5, s=30, edgecolor='k', linewidth=0.5)

            # Raw null pmf itself (not yet cumulative), over v — the
            # building block every other panel derives from.
            ax_pmf.step(v_vals[lo:hi + 1], pmf[lo:hi + 1], where='mid', color=color, label=label)

            # Same null pmf/cdf, but expressed as the CMVP z-score/effect
            # size instead of pushing v through the Jaccard map — puts
            # every pair on a common, comparable scale.
            z_vals = (v_vals - mean_v) / stat_denom if stat_denom > 0 else np.zeros_like(v_vals, dtype=float)
            cn_cross_idx = min(cross_idx, len(z_vals) - 1)
            ax_cn.step(z_vals, cdf, where='post', color=color, label=label_main)
            ax_cn.scatter([z_vals[cn_cross_idx]], [0.5], color=color, zorder=5, s=30, edgecolor='k', linewidth=0.5)

            # Track the z-range that actually carries visible probability mass,
            # so one pair's long discrete tail doesn't stretch the shared
            # x-axis and squash everyone else's curve near zero.
            z_abs_max = max(z_abs_max, abs(z_vals[lo]), abs(z_vals[hi]))

            # Same v-support, but J and Z plotted against each other directly:
            # shows the value-level mapping (not the CDF) between the two
            # similarity notions for this pair — the source of "same J,
            # different meaning" when this curve differs across pairs.
            # Full (untrimmed) data, matching how panels 1/2 draw their full
            # step functions and rely on axis limits (not data slicing) to
            # crop the view — keeps all three panels' visible ranges consistent.
            ax_jz.plot(j_vals, z_vals, color=color, label=label_main)

        ax.axhline(0.5, color='k', lw=0.8, ls='--', alpha=0.5)
        ax.set_xlim(0, j_axis_max if j_axis_max > 0 else 1)
        if normalize_jaccard:
            ax.set_xlabel('Normalized Jaccard similarity $J_{ij} / J_{max}$')
            ax.set_title('Jaccard similarity (normalized)')
        else:
            ax.set_xlabel('Jaccard similarity $J_{ij}$')
            ax.set_title('Jaccard similarity (shared scale)')
        ax.set_ylabel('Null CDF $P(J \\leq j)$')
        ax.legend(fontsize=PlotConfig.FONT_SIZE_TICK, loc='lower right')
        ax.grid(alpha=PlotConfig.GRID_ALPHA)

        if stat == 'zscore':
            stat_label = 'CMVP z-score'
            stat_expr = '$(CN_{ij} - E[CN_{ij}]) / \\mathrm{std}[CN_{ij}]$'
            stat_symbol = 'Z'
        else:
            stat_label = 'CMVP effect size'
            stat_expr = '$(CN_{ij} - E[CN_{ij}]) / \\mathrm{Var}[CN_{ij}]$'
            stat_symbol = 'E'

        ax_cn.axhline(0.5, color='k', lw=0.8, ls='--', alpha=0.5)
        ax_cn.axvline(0, color='k', lw=0.8, ls=':', alpha=0.4)
        if z_abs_max > 0:
            ax_cn.set_xlim(-z_abs_max, z_abs_max)
        ax_cn.set_xlabel(f'{stat_label} {stat_expr}')
        ax_cn.set_ylabel(f'Null CDF $P({stat_symbol} \\leq {stat_symbol.lower()})$')
        ax_cn.set_title(f'CMVP similarity ({"z-score" if stat == "zscore" else "effect size"})')
        ax_cn.grid(alpha=PlotConfig.GRID_ALPHA)

        ax_jz.axhline(0, color='k', lw=0.8, ls=':', alpha=0.4)
        if z_abs_max > 0:
            ax_jz.set_ylim(-z_abs_max, z_abs_max)
        ax_jz.set_xlim(0, j_axis_max if j_axis_max > 0 else 1)
        ax_jz.set_xlabel('Normalized Jaccard $J_{ij} / J_{max}$' if normalize_jaccard else 'Jaccard $J_{ij}$')
        ax_jz.set_ylabel(stat_label)
        ax_jz.set_title(f'Jaccard vs. {stat_label}')
        ax_jz.legend(fontsize=PlotConfig.FONT_SIZE_TICK, loc='upper left')
        ax_jz.grid(alpha=PlotConfig.GRID_ALPHA)

        if ax_fit is not None:
            ax_fit.set_yscale('log')
            ax_fit.set_xlabel('Degree $k$')
            ax_fit.set_ylabel('Fitted fitness $x$')
            ax_fit.set_title('Degree vs. fitted fitness parameter')
            ax_fit.legend(fontsize=PlotConfig.FONT_SIZE_TICK, loc='upper left')
            ax_fit.grid(alpha=PlotConfig.GRID_ALPHA)

        ax_pmf.set_xlabel('Common neighbors $CN_{ij}$')
        ax_pmf.set_ylabel('Null pmf $P(CN_{ij} = v)$')
        ax_pmf.set_title('Null probability mass function by degree pair')
        ax_pmf.legend(fontsize=PlotConfig.FONT_SIZE_TICK, loc='upper right')
        ax_pmf.grid(alpha=PlotConfig.GRID_ALPHA)

        fig.suptitle('CMVP and Jaccard Relationship by Degree', **PlotConfig.SUPTITLE_KW)
        fig.tight_layout()
        _finalize_plot(fig, save_path, show)

        fig2.suptitle('Null model diagnostics by degree pair', **PlotConfig.SUPTITLE_KW)
        fig2.tight_layout()
        save_path2 = None
        if save_path is not None:
            root, ext = os.path.splitext(save_path)
            save_path2 = f'{root}_degree_pmf{ext}'
        _finalize_plot(fig2, save_path2, show)

    def plot_zscore_vs_degree_product(self, stat: str = 'zscore', color_by: str = 'exp_cn',
                                      x_axis: str = 'geomean',
                                      save_path: Optional[str] = None, show: bool = True):
        """
        Scatter plot of a degree-based x-axis against CN z-score or effect size
        (y-axis), colored by E[CN] or std[CN].

        Checks whether the null model's degree correction actually removed degree
        bias from significance: if CMVP is well-calibrated, the y-axis should show
        no systematic trend against the x-axis. A visible slope means pairs at that
        end of the x-axis are getting systematically inflated or deflated
        significance/effect. Only pairs with CN > 0 are included.

        Parameters
        ----------
        stat : str
            'zscore' (default) or 'effect' — which statistic to put on the
            y-axis (see _stat_matrix()).
        color_by : str
            'exp_cn' (default) — E[CN] under the null model;
            'std'              — std[CN] under the null model;
            'cn'               — observed common neighbors.
        x_axis : str
            'geomean' (default) — degree geomean sqrt(k_i*k_j), a function of i
            and j alone;
            'third_party_degree' — the p_ik*p_jk-weighted mean degree of the
            third-party nodes k that actually drive E[CN]_ij:
            sum_k(k_k*p_ik*p_jk) / E[CN]_ij. Unlike geomean, this depends on the
            full degree sequence (all k, not just i and j), so it can separate
            pairs with identical degree geomean but very different expected
            overlap because they route through different-degree third parties
            (e.g. a shared hub vs. many diffuse low-degree neighbors).
        """
        deg = np.asarray(self.cmvp.A_bin.sum(axis=1)).flatten().astype(float)
        n = self.cmvp.n
        obs_mat = self.cmvp.obs_sim_matrix
        exp_mat = self.cmvp.exp_sim_matrix
        std_mat = self.cmvp.exp_sim_matrix_std
        stat_mat, stat_label = self._stat_matrix(stat)

        if x_axis == 'third_party_degree':
            P = self.cmvp.p_matrix
            if P is None:
                raise RuntimeError("Call fit_configuration_model() first to compute p_matrix.")
            weighted_deg_sum = P @ (P * deg[:, np.newaxis])
            np.fill_diagonal(weighted_deg_sum, 0)
            with np.errstate(invalid='ignore', divide='ignore'):
                x_mat = np.where(exp_mat > 0, weighted_deg_sum / exp_mat, 0.0)
            x_title = 'effective third-party degree'
            x_label = r'$\bar{k}_{ij} = \sum_k k_k\,p_{ik}p_{jk}\, /\, E[CN]_{ij}$'
        else:
            x_mat = np.sqrt(np.outer(deg, deg))
            x_title = 'degree geomean'
            x_label = r'degree geomean $\sqrt{k_i k_j}$'

        if self.directed:
            ii, jj = np.where(~np.eye(n, dtype=bool))
        else:
            ii, jj = np.triu_indices(n, k=1)

        cn_flat = obs_mat[ii, jj]
        stat_vals = stat_mat[ii, jj]

        valid = ~np.isnan(stat_vals) & (cn_flat > 0)
        stat_v = stat_vals[valid]
        exp_cn_v = exp_mat[ii, jj][valid]
        std_v = std_mat[ii, jj][valid] if std_mat is not None else np.zeros(len(stat_v))
        x_v = x_mat[ii, jj][valid]

        cn_v = cn_flat[valid]
        color_vals, color_label = self._color_by_std_cn_expcn(color_by, std_v, cn_v, exp_cn_v)

        fig, ax = plt.subplots(figsize=PlotConfig.SMALL)
        sc = ax.scatter(x_v, stat_v, c=color_vals, cmap='viridis', s=30,
                        alpha=PlotConfig.SCATTER_ALPHA, linewidth=0, rasterized=True)
        plt.colorbar(sc, ax=ax, label=color_label)

        ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)

        pear_r, _ = pearsonr(x_v, stat_v)
        spear_r, _ = spearmanr(x_v, stat_v)
        ax.text(0.02, 0.98, f'Pearson r = {pear_r:.3f}\nSpearman ρ = {spear_r:.3f}',
                transform=ax.transAxes, ha='left', va='top', fontsize=PlotConfig.FONT_SIZE_ANNOTATION,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        _configure_scatter_plot(ax, 'Degree Bias Check',
                                x_label,
                                stat_label)
        _finalize_plot(fig, save_path, show)

    def plot_metric_correlation(self, method: str = 'both', save_path: Optional[str] = None,
                                show: bool = True):
        """
        Correlation matrix heatmap for CN, Jaccard, Resource Allocation, and z-score.
        Only pairs with CN > 0 are included.

        Parameters
        ----------
        method : str
            'both' (default) — side-by-side Spearman and Pearson panels;
            'rank'            — Spearman rank correlation only;
            'linear'          — Pearson linear correlation only.
        """
        if method not in ('both', 'rank', 'linear'):
            raise ValueError("method must be 'both', 'rank', or 'linear'")
        deg = np.asarray(self.cmvp.A_bin.sum(axis=1)).flatten()
        n = self.cmvp.n
        obs_mat = self.cmvp.obs_sim_matrix
        exp_mat = self.cmvp.exp_sim_matrix
        std_mat = self.cmvp.exp_sim_matrix_std
        effect_mat = self.cmvp.test_sim_matrix_effect

        if self.directed:
            ii, jj = np.where(~np.eye(n, dtype=bool))
        else:
            ii, jj = np.triu_indices(n, k=1)

        cn_flat = obs_mat[ii, jj] if obs_mat is not None else np.zeros(len(ii))

        with np.errstate(invalid='ignore', divide='ignore'):
            union = deg[ii] + deg[jj] - cn_flat
            jaccard_vals = np.where(union > 0, cn_flat / union, np.nan)
            if obs_mat is not None and exp_mat is not None and std_mat is not None:
                z_vals = self._stat_matrix('zscore')[0][ii, jj]
            else:
                z_vals = np.zeros(len(ii))

        A = self.cmvp.A_bin.astype(float)
        A_dense = np.asarray(A.todense()) if hasattr(A, 'todense') else A
        ra_vals = np.full(len(ii), np.nan)
        for k, (i, j) in enumerate(zip(ii, jj)):
            shared = np.where((A_dense[i] > 0) & (A_dense[j] > 0))[0]
            if len(shared) == 0:
                continue
            d = deg[shared].astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                ra_vals[k] = np.sum(1.0 / np.where(d > 0, d, np.nan))

        exp_cn_vals = exp_mat[ii, jj] if exp_mat is not None else np.zeros(len(ii))
        effect_vals = effect_mat[ii, jj] if effect_mat is not None else np.full(len(ii), np.nan)

        tsm = self.cmvp.test_sim_matrix_sig
        if tsm is not None:
            _tsm = tsm.toarray() if hasattr(tsm, 'toarray') else tsm
            cmvp_sim_vals = _tsm[ii, jj]
        else:
            cmvp_sim_vals = np.full(len(ii), np.nan)

        valid = (cn_flat > 0) & ~np.isnan(jaccard_vals) & ~np.isnan(z_vals) & ~np.isnan(ra_vals) & \
                ~np.isnan(exp_cn_vals) & ~np.isnan(effect_vals)

        data = np.column_stack([
            cn_flat[valid],
            ra_vals[valid],
            jaccard_vals[valid],
            z_vals[valid],
            effect_vals[valid],
            cmvp_sim_vals[valid],
            exp_cn_vals[valid],
        ])
        labels = ['CN', 'Resource Alloc', 'Jaccard', 'CMVP z-score', 'CMVP Effect', 'CMVP Sig.',
                 'E[CN]']

        panels = []
        if method in ('both', 'rank'):
            spearman_corr = np.array([[spearmanr(data[:, i], data[:, j])[0]
                                       for j in range(data.shape[1])]
                                      for i in range(data.shape[1])])
            panels.append((spearman_corr, 'Spearman ρ', 'Rank correlation (Spearman)'))
        if method in ('both', 'linear'):
            pearson_corr = np.corrcoef(data, rowvar=False)
            panels.append((pearson_corr, 'Pearson r', 'Linear correlation (Pearson)'))

        figsize = PlotConfig.TWO_PANEL if len(panels) == 2 else (PlotConfig.TWO_PANEL[0] / 2, PlotConfig.TWO_PANEL[1])
        fig, axes = plt.subplots(1, len(panels), figsize=figsize)
        if len(panels) == 1:
            axes = [axes]
        for ax, (corr, cbar_label, title) in zip(axes, panels):
            im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            fig.colorbar(im, ax=ax, label=cbar_label)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha='right')
            ax.set_yticklabels(labels)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                            fontsize=PlotConfig.FONT_SIZE_ANNOTATION, color='black')
            ax.set_title(title)
        fig.suptitle('Pair-level correlations  (CN > 0)', **PlotConfig.SUPTITLE_KW)
        _finalize_plot(fig, save_path, show)

    def plot_zscore_vs_effect(self, mask_zero_std: bool = True, color_by: str = 'std',
                              save_path: Optional[str] = None, show: bool = True):
        """
        Scatter of effect size theta-hat (x-axis) against z-score (y-axis),
        colored by std[CN] under the null (default) or observed CN.

        theta-hat = (obs - exp) / var and z-score = (obs - exp) / std, so
        z = theta-hat * std: for a fixed effect size, pairs with larger null
        std (typically higher-degree pairs, more "opportunity"/evidence) get
        pulled to larger |z| even though the underlying association strength
        (theta-hat) is unchanged. Color reveals this directly — points at the
        same x (effect) but different y (z) are separated by std, i.e. by how
        much evidence backed the same effect size, not by how strong the tie
        actually is.

        Parameters
        ----------
        color_by : str
            'std' (default) — std[CN] under the null model;
            'cn'             — observed CN.
        """
        effect_mat = self.cmvp.test_sim_matrix_effect
        z_mat = self.cmvp.test_sim_matrix_zscore
        std_mat = self.cmvp.exp_sim_matrix_std
        obs_mat = self.cmvp.obs_sim_matrix
        if effect_mat is None or z_mat is None or std_mat is None:
            raise RuntimeError("Call validate_projection() first.")

        n = self.cmvp.n
        if self.directed:
            ii, jj = np.where(~np.eye(n, dtype=bool))
        else:
            ii, jj = np.triu_indices(n, k=1)

        effect_vals = effect_mat[ii, jj]
        z_vals = z_mat[ii, jj]
        std_vals = std_mat[ii, jj]
        cn_vals = obs_mat[ii, jj]

        if mask_zero_std:
            mask = std_vals > 0
            effect_vals, z_vals, std_vals, cn_vals = \
                effect_vals[mask], z_vals[mask], std_vals[mask], cn_vals[mask]

        if color_by == 'cn':
            color_vals, color_label = cn_vals, 'CN'
        else:
            color_vals, color_label = std_vals, 'std[CN] (null model)'

        rho, _ = spearmanr(effect_vals, z_vals)
        r = np.corrcoef(effect_vals, z_vals)[0, 1]

        fig, ax = plt.subplots(figsize=PlotConfig.SQUARE)
        scatter_kw = dict(cmap='viridis', s=12, alpha=PlotConfig.SCATTER_ALPHA,
                          linewidth=0, rasterized=True)
        sc = ax.scatter(effect_vals, z_vals, c=color_vals, **scatter_kw)
        fig.colorbar(sc, ax=ax, label=color_label)

        ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        ax.text(0.02, 0.97, rf'Spearman $\rho$ = {rho:.3f}' + '\n' + rf'Pearson $r$ = {r:.3f}',
                transform=ax.transAxes, ha='left', va='top', fontsize=PlotConfig.FONT_SIZE_ANNOTATION,
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))

        _configure_scatter_plot(ax, 'Effect size vs z-score',
                                r'$\widehat{\mathrm{Effect}}$ = (obs - exp) / var',
                                'z-score  (obs − exp) / std')
        _finalize_plot(fig, save_path, show)

