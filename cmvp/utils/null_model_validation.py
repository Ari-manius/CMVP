"""
CMVP Null Model Validation - Lightweight
=========================================

Analyzes and visualizes quality of fitted configuration models.

Classes:
    NullModelValidator: Validates fitted configuration model

Functions:
    validate_null_model: Quick validation wrapper
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import networkx as nx

from .validation import BaseValidator, PlotConfig, _finalize_plot, _configure_scatter_plot, ecdf_xy


class NullModelValidator(BaseValidator):
    """
    Validates quality of a fitted configuration model.

    Analyzes:
    - Degree preservation (topology)
    - Strength preservation (weights, if applicable)
    - Edge probability distribution

    Parameters
    ----------
    cmvp : CMVP
        Fitted CMVP object
    """

    def __init__(self, cmvp):
        """Initialize with fitted CMVP object."""
        super().__init__(cmvp)

        if self.p_matrix is None:
            raise ValueError("p_matrix not found. Call cmvp.fit_configuration_model() first.")

        self._results = None

    def validate(self, plot: bool = True, verbose: bool = True,
                save_path: Optional[str] = None, show: bool = True) -> Dict:
        """
        Run full null model validation.

        Parameters
        ----------
        plot : bool
            Whether to create diagnostic plots
        verbose : bool
            Whether to print progress messages
        save_path, show
            Forwarded to plot() when plot=True.

        Returns
        -------
        dict
            Dictionary with degree, strength (if applicable), and edge_probs keys
        """
        results = {}

        if verbose:
            print(f"\n{'='*70}")
            print("NULL MODEL VALIDATION")
            print(f"{'='*70}\n")

        # 1. Degree preservation
        if verbose:
            print("Analyzing DEGREE preservation (topology)...")
        results['degree'] = self._analyze_degree()

        # 2. Strength preservation (if weighted)
        if self.weighted:
            if verbose:
                print("Analyzing STRENGTH preservation (weights)...")
            results['strength'] = self._analyze_strength()
        else:
            results['strength'] = None

        # 3. Edge probability distribution
        if verbose:
            print("Analyzing edge probability distribution...")
        results['edge_probs'] = self._analyze_edge_probs()

        self._results = results

        if verbose:
            self._print_summary(results)

        if plot:
            self.plot(results, save_path=save_path, show=show)

        return results

    def _analyze_degree(self) -> Dict:
        """Analyze degree preservation (topology)."""
        A_binary = (self.A > 0).astype(int)
        p_matrix = self.p_matrix

        if self.directed:
            obs_out = A_binary.sum(axis=1)
            obs_in = A_binary.sum(axis=0)
            exp_out = p_matrix.sum(axis=1)
            exp_in = p_matrix.sum(axis=0)

            corr_out = np.corrcoef(obs_out, exp_out)[0, 1]
            corr_in = np.corrcoef(obs_in, exp_in)[0, 1]

            return {
                'obs_out': obs_out, 'exp_out': exp_out, 'corr_out': corr_out,
                'obs_in': obs_in, 'exp_in': exp_in, 'corr_in': corr_in
            }
        else:
            obs_deg = A_binary.sum(axis=1)
            exp_deg = p_matrix.sum(axis=1)
            corr = np.corrcoef(obs_deg, exp_deg)[0, 1]

            return {
                'obs': obs_deg, 'exp': exp_deg, 'correlation': corr,
                'rmse': np.sqrt(np.mean((obs_deg - exp_deg)**2))
            }

    def _analyze_strength(self) -> Dict:
        """Analyze strength preservation (weights) from null model."""
        obs_str = self.A.sum(axis=1)

        # Try to get expected weights from null model
        # NEMtropy stores expected weights for ECM (Exponential Configuration Model)
        exp_str = None

        # Try different sources for expected weights
        if hasattr(self.cmvp, 'w_matrix') and self.cmvp.w_matrix is not None:
            w_mat = self.cmvp.w_matrix
            if hasattr(w_mat, 'toarray'):
                w_mat = w_mat.toarray()
            exp_str = w_mat.sum(axis=1)
        elif hasattr(self.cmvp, 'null_model') and hasattr(self.cmvp.null_model, 'expected_weights'):
            w_mat = self.cmvp.null_model.expected_weights
            if hasattr(w_mat, 'toarray'):
                w_mat = w_mat.toarray()
            exp_str = w_mat.sum(axis=1)
        elif hasattr(self.cmvp, 'expected_weights') and self.cmvp.expected_weights is not None:
            w_mat = self.cmvp.expected_weights
            if hasattr(w_mat, 'toarray'):
                w_mat = w_mat.toarray()
            exp_str = w_mat.sum(axis=1)

        if exp_str is None:
            return {}

        if self.directed:
            obs_out = obs_str
            obs_in = self.A.sum(axis=0)
            exp_out = exp_str
            exp_in = np.asarray(w_mat.sum(axis=0)).flatten()

            corr_out = np.corrcoef(obs_out, exp_out)[0, 1]
            corr_in = np.corrcoef(obs_in, exp_in)[0, 1]

            return {
                'obs_out': obs_out, 'exp_out': exp_out, 'corr_out': corr_out,
                'obs_in': obs_in, 'exp_in': exp_in, 'corr_in': corr_in
            }
        else:
            corr = np.corrcoef(obs_str, exp_str)[0, 1]
            rmse = np.sqrt(np.mean((obs_str - exp_str)**2))

            return {'obs': obs_str, 'exp': exp_str, 'correlation': corr, 'rmse': rmse}

    def _analyze_edge_probs(self) -> Dict:
        """Analyze edge probability distribution."""
        probs = self._extract_offdiag_values(self.p_matrix)
        return {
            'probs': probs,
            'mean': np.mean(probs),
            'median': np.median(probs),
            'std': np.std(probs)
        }

    def _print_summary(self, results: Dict):
        """Print validation summary."""
        deg = results['degree']
        corr_key = 'correlation' if 'correlation' in deg else 'corr_out'

        if deg[corr_key] > 0.99:
            print(f"  ✓ Degree correlation: {deg[corr_key]:.4f} (EXCELLENT)\n")
        elif deg[corr_key] > 0.95:
            print(f"  ✓ Degree correlation: {deg[corr_key]:.4f} (GOOD)\n")
        else:
            print(f"  ⚠ Degree correlation: {deg[corr_key]:.4f} (POOR)\n")

        # Check if strength results exist and are not empty
        if results['strength'] is not None and len(results['strength']) > 0:
            strength = results['strength']
            corr_key = 'correlation' if 'correlation' in strength else 'corr_out'
            if corr_key in strength:
                status = "EXCELLENT" if strength[corr_key] > 0.99 else "GOOD" if strength[corr_key] > 0.95 else "POOR"
                print(f"  {'✓' if strength[corr_key] > 0.95 else '⚠'} Strength correlation: {strength[corr_key]:.4f} ({status})\n")

    def plot(self, results: Optional[Dict] = None, save_path: Optional[str] = None, show: bool = True):
        """Create diagnostic plots for null model."""
        if results is None:
            results = self._results
        if results is None:
            print("No results to plot. Run validate() first.")
            return

        # Determine number of subplots needed
        has_strength = results['strength'] is not None and len(results['strength']) > 0
        has_x = hasattr(self.cmvp, 'x') and self.cmvp.x is not None
        n_panels = (4 if has_strength else 3) + (1 if has_x else 0)

        fig, axes = plt.subplots(1, n_panels, figsize=(5.3 * n_panels, 5))

        def _log_refline(ax, *arrays):
            """Draw y=x reference line starting at 10^0, shared max on both axes."""
            vals = np.concatenate([np.asarray(a).ravel() for a in arrays])
            vals = vals[vals > 0]
            hi = vals.max() * 1.2 if len(vals) else 10
            ax.plot([1, hi], [1, hi], 'r--', lw=PlotConfig.LINE_WIDTH, label='Perfect')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(1, hi)
            ax.set_ylim(1, hi)

        # 1. Degree preservation
        deg = results['degree']
        if 'obs' in deg:
            # Undirected graph
            obs, exp = deg['obs'], deg['exp']
            axes[0].scatter(obs, exp, alpha=PlotConfig.SCATTER_ALPHA, s=40, color=PlotConfig.PRIMARY)
            _log_refline(axes[0], obs, exp)
            _configure_scatter_plot(axes[0], f'Degree Preservation (r={deg["correlation"]:.4f})',
                                   'Observed Degree', 'Expected Degree')
        elif 'obs_out' in deg:
            # Directed graph - plot both out-degree and in-degree
            obs_out, exp_out = deg['obs_out'], deg['exp_out']
            obs_in, exp_in = deg['obs_in'], deg['exp_in']
            axes[0].scatter(obs_out, exp_out, alpha=PlotConfig.SCATTER_ALPHA, s=40, color=PlotConfig.PRIMARY, label='Out-degree')
            axes[0].scatter(obs_in, exp_in, alpha=PlotConfig.SCATTER_ALPHA, s=40, color=PlotConfig.SECONDARY, label='In-degree')
            _log_refline(axes[0], obs_out, exp_out, obs_in, exp_in)
            title = f'Degree Preservation (r_out={deg["corr_out"]:.4f}, r_in={deg["corr_in"]:.4f})'
            _configure_scatter_plot(axes[0], title, 'Observed Degree', 'Expected Degree')

        # 1a. Strength preservation (if weighted)
        if has_strength:
            strength = results['strength']
            if 'obs' in strength:
                # Undirected weighted graph
                obs, exp = strength['obs'], strength['exp']
                axes[1].scatter(obs, exp, alpha=PlotConfig.SCATTER_ALPHA, s=40, color=PlotConfig.SECONDARY)
                _log_refline(axes[1], obs, exp)
                _configure_scatter_plot(axes[1], f'Strength Preservation (r={strength["correlation"]:.4f})',
                                       'Observed Strength', 'Expected Strength')
            elif 'obs_out' in strength:
                # Directed weighted graph
                obs_out, exp_out = strength['obs_out'], strength['exp_out']
                obs_in, exp_in = strength['obs_in'], strength['exp_in']
                axes[1].scatter(obs_out, exp_out, alpha=PlotConfig.SCATTER_ALPHA, s=40, color=PlotConfig.SECONDARY, label='Out-strength')
                axes[1].scatter(obs_in, exp_in, alpha=PlotConfig.SCATTER_ALPHA, s=40, color='orange', label='In-strength')
                _log_refline(axes[1], obs_out, exp_out, obs_in, exp_in)
                title = f'Strength Preservation (r_out={strength["corr_out"]:.4f}, r_in={strength["corr_in"]:.4f})'
                _configure_scatter_plot(axes[1], title, 'Observed Strength', 'Expected Strength')

            edge_prob_idx = 2
        else:
            edge_prob_idx = 1

        # 2. Edge probability distribution
        probs = results['edge_probs']['probs']
        counts, bins = np.histogram(probs, bins=50)
        axes[edge_prob_idx].bar((bins[:-1] + bins[1:]) / 2, counts, width=bins[1]-bins[0], alpha=0.7,
                               edgecolor='black', color=PlotConfig.PRIMARY)
        axes[edge_prob_idx].axvline(results['edge_probs']['mean'], color=PlotConfig.DANGER,
                                   linestyle='--', lw=PlotConfig.LINE_WIDTH,
                                   label=f"Mean={results['edge_probs']['mean']:.3f}")
        axes[edge_prob_idx].set_xlabel('Edge Probability', fontweight='bold')
        axes[edge_prob_idx].set_ylabel('Count', fontweight='bold')
        axes[edge_prob_idx].set_title('Edge Probability Distribution', fontweight='bold')
        axes[edge_prob_idx].legend()
        axes[edge_prob_idx].grid(alpha=PlotConfig.GRID_ALPHA)

        # 3. Edge probability vs degree product
        deg_idx = edge_prob_idx + 1
        A_binary = (self.A > 0).astype(int)
        deg = A_binary.sum(axis=1)
        p_matrix = self.p_matrix
        iu, ju = np.triu_indices(len(deg), k=1)
        deg_product = (deg[iu] * deg[ju]).astype(float)
        p_offdiag = np.asarray(p_matrix)[iu, ju]

        axes[deg_idx].scatter(deg_product, p_offdiag, alpha=PlotConfig.SCATTER_ALPHA, s=15,
                              color=PlotConfig.PRIMARY, linewidth=0)
        axes[deg_idx].set_xscale('log')
        _configure_scatter_plot(axes[deg_idx], 'Edge Probability vs Degree Product',
                               'Degree Product  $k_i k_j$', 'Edge Probability $p_{ij}$')

        # 4. Degree vs fitted fitness parameter (undirected unweighted models only)
        if has_x:
            fit_idx = deg_idx + 1
            x_real = np.asarray(self.cmvp.x, dtype=float)
            pos = deg > 0
            axes[fit_idx].scatter(deg[pos], x_real[pos], alpha=PlotConfig.SCATTER_ALPHA, s=25,
                                  color=PlotConfig.PRIMARY, linewidth=0)
            axes[fit_idx].set_yscale('log')
            _configure_scatter_plot(axes[fit_idx], 'Degree vs Fitted Fitness',
                                   'Degree $k$', 'Fitted Fitness $x$')

        _finalize_plot(fig, save_path, show)

    def plot_std_vs_expected_cn(self, mask_zero_exp: bool = True,
                                save_path: Optional[str] = None, show: bool = True):
        """
        Left: scatter of E[CN] (x-axis) against std[CN] (y-axis) under the null
        model, with a twin y-axis showing the per-pair Fano factor
        (Var[CN]/E[CN]).

        Since Var(CN) = E[CN] - sum_k (p_ik*p_jk)^2 <= E[CN], std[CN] is always
        <= sqrt(E[CN]) (the Poisson reference, dashed) and the Fano factor is
        always <= 1. Pairs near the dashed curve / Fano ~ 1 behave like Poisson
        (low individual connection probabilities); pairs falling below /
        Fano << 1 are under-dispersed, typically high-degree/hub pairs where
        some p_ik*p_jk approach 1.

        Right: empirical CDF of the Fano factor across all tested pairs, a
        global summary of how well the Poisson approximation fits.

        Requires validate_projection() to have been run (needs exp_sim_matrix /
        exp_sim_matrix_std, not just the fitted p_matrix).
        """
        exp_mat = self.cmvp.exp_sim_matrix
        std_mat = self.cmvp.exp_sim_matrix_std
        if exp_mat is None or std_mat is None:
            raise RuntimeError("Call validate_projection() first.")

        n = self.cmvp.n
        if self.directed:
            ii, jj = np.where(~np.eye(n, dtype=bool))
        else:
            ii, jj = np.triu_indices(n, k=1)

        exp_vals = exp_mat[ii, jj]
        std_vals = std_mat[ii, jj]

        if mask_zero_exp:
            mask = exp_vals > 0
            exp_vals, std_vals = exp_vals[mask], std_vals[mask]

        fano = np.divide(std_vals ** 2, exp_vals, out=np.zeros_like(exp_vals), where=exp_vals > 0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PlotConfig.WIDE)

        # 1. std[CN] vs E[CN]
        ax1.scatter(exp_vals, std_vals, s=10, alpha=PlotConfig.SCATTER_ALPHA,
                   color='steelblue', linewidth=0, rasterized=True)
        x_line = np.linspace(0, exp_vals.max() if len(exp_vals) else 1, 200)
        ax1.plot(x_line, np.sqrt(x_line), 'k--', lw=1, label='Poisson: std = √E[CN]')
        ax1.legend(fontsize=PlotConfig.FONT_SIZE_TICK)
        _configure_scatter_plot(ax1, 'std[CN] vs E[CN]  (null model)', 'E[CN]', 'std[CN]')

        # 2. CDF of the Fano factor
        sorted_fano, cdf = ecdf_xy(fano)
        ax2.plot(sorted_fano, cdf, color=PlotConfig.PRIMARY, lw=2)
        ax2.axvline(1.0, color='black', linestyle=':', lw=1, label='Poisson (F=1)')
        mean_fano = fano.mean() if len(fano) else float('nan')
        ax2.axvline(mean_fano, color=PlotConfig.DANGER, linestyle='--', lw=1.5,
                   label=f'Mean F={mean_fano:.2f}')
        ax2.set_xlim(0, 1.05)
        ax2.set_ylim(0, 1.02)
        ax2.legend(fontsize=PlotConfig.FONT_SIZE_TICK)
        ax2.set_xlabel('Fano factor  Var[CN]/E[CN]', fontweight='bold')
        ax2.set_ylabel('Cumulative fraction of pairs', fontweight='bold')
        ax2.set_title('CDF of Fano factor', fontweight='bold')
        ax2.grid(alpha=PlotConfig.GRID_ALPHA)

        _finalize_plot(fig, save_path, show)



def validate_null_model(cmvp, plot: bool = True, verbose: bool = True,
                        save_path: Optional[str] = None, show: bool = True) -> NullModelValidator:
    """
    Validate null model quality.

    Parameters
    ----------
    cmvp : CMVP
        Fitted CMVP object
    plot : bool
        Whether to create plots
    verbose : bool
        Whether to print progress messages
    save_path, show
        Forwarded to NullModelValidator.plot() when plot=True.

    Returns
    -------
    NullModelValidator
        Validator object with results
    """
    validator = NullModelValidator(cmvp)
    validator.validate(plot=plot, verbose=verbose, save_path=save_path, show=show)
    return validator
