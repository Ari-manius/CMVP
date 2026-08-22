"""
CMVP Backbone Validation
=========================

Analyzes and visualizes quality of extracted backbone projections.

Split across three files to keep each under ~1000 lines:
    backbone_validation.py        - BackboneValidator (this file): construction,
                                     validate()/plot() orchestration, shared helpers.
    backbone_validation_nullcal.py - _NullCalibrationMixin: null-model calibration
                                     and degree-bias diagnostic plots.
    backbone_validation_similarity.py - _SimilarityPlotsMixin: similarity-measure
                                     comparison plots (heatmap, dendrogram, network).

Classes:
    BackboneValidator: Validates extracted backbone (composes both mixins below).

Functions:
    validate_backbone: Quick validation wrapper
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from typing import Dict, Optional

from .validation import BaseValidator, PlotConfig, _finalize_plot
from .backbone_validation_nullcal import _NullCalibrationMixin
from .backbone_validation_similarity import _SimilarityPlotsMixin


class BackboneValidator(_NullCalibrationMixin, _SimilarityPlotsMixin, BaseValidator):
    """
    Validates quality of extracted backbone.

    Analyzes:
    - Backbone density vs alpha threshold
    - P-value distribution
    - Observed vs expected similarity
    - P-value vs expected similarity

    Parameters
    ----------
    cmvp : CMVP
        Fitted CMVP object
    backbone : sparse or networkx.Graph
        Extracted backbone
    pvalues : np.ndarray
        P-value matrix

    Notes
    -----
    Most diagnostic plot_* methods live on the two mixins this class composes
    (see module docstring) rather than directly on this class.
    """

    def __init__(self, cmvp):
        """Initialize backbone validator."""
        super().__init__(cmvp)
        self.alpha = cmvp._config.get('alpha') or 0.05
        self.correction = cmvp._config.get('correction_method') or 'fdr'
        self.obs_sim = getattr(cmvp, 'obs_sim_matrix', None)
        self.exp_sim = getattr(cmvp, 'exp_sim_matrix', None)
        self._results = None

    @property
    def backbone(self):
        return self.cmvp.backbone

    @property
    def pvalues(self):
        return self.cmvp.test_sim_matrix_pvalue

    def _get_tested_pvalues(self) -> np.ndarray:
        """Return p-values for tested pairs only."""
        return self.cmvp._pvalues_array

    def _stat_matrix(self, stat: str):
        """
        Dense (obs - exp) / std [zscore] or (obs - exp) / var [effect] matrix,
        with the corresponding axis/colorbar label. Shared by the various
        BackboneValidator plots that let the caller pick which of the two to
        use — significance (zscore) conflates association strength with
        amount of evidence; effect size (theta-hat) has the degree/opportunity
        term divided out.
        """
        if stat not in ('zscore', 'effect', 'cn'):
            raise ValueError(f"stat must be 'zscore', 'effect' or 'cn', got '{stat}'")

        obs_mat = self.cmvp.obs_sim_matrix
        exp_mat = self.cmvp.exp_sim_matrix
        std_mat = self.cmvp.exp_sim_matrix_std
        if obs_mat is None or exp_mat is None or std_mat is None:
            raise RuntimeError("Call validate_projection() first.")

        if stat == 'cn':
            return obs_mat.copy(), 'CN'

        with np.errstate(invalid='ignore', divide='ignore'):
            if stat == 'zscore':
                mat = np.where(std_mat > 0, (obs_mat - exp_mat) / std_mat, np.nan)
                label = 'z-score  (obs − exp) / std'
            else:
                null_var = std_mat ** 2
                mat = np.where(null_var > 0, (obs_mat - exp_mat) / null_var, np.nan)
                label = r'$\widehat{\mathrm{Effect}}$ = (obs − exp) / var'

        return mat, label

    @staticmethod
    def _color_by_std_cn_expcn(color_by: str, std_vals, cn_vals, exp_cn_vals):
        """Shared 'std'/'cn'/default-exp_cn dispatch for scatter-plot coloring."""
        if color_by == 'std':
            return std_vals, 'std[CN]'
        elif color_by == 'cn':
            return cn_vals, 'CN'
        else:
            return exp_cn_vals, 'E[CN]'

    def validate(self, plot: bool = True, verbose: bool = True,
                save_path: Optional[str] = None, show: bool = True) -> Dict:
        """
        Run full backbone validation.

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
            Dictionary with density, pvalue_dist, similarity, and pval_vs_pred keys
        """
        if verbose:
            print(f"\n{'='*70}")
            print("BACKBONE VALIDATION")
            print(f"{'='*70}\n")

        results = {}

        # 1. Density vs alpha
        if verbose:
            print("Analyzing backbone density vs alpha...")
        results['density'] = self._analyze_density()

        # 2. P-value distribution
        if verbose:
            print("Analyzing p-value distribution...")
        pval_tested = self._get_tested_pvalues()
        results['pvalue_dist'] = {
            'mean': np.mean(pval_tested),
            'median': np.median(pval_tested),
            'std': np.std(pval_tested)
        }

        # 3. Similarity comparison
        if self.obs_sim is not None and self.exp_sim is not None:
            if verbose:
                print("Analyzing observed vs expected similarity...")
            results['similarity'] = self._analyze_similarity()
        else:
            results['similarity'] = None

        self._results = results

        if verbose:
            self._print_summary(results)

        if plot:
            self.plot(results, save_path=save_path, show=show)

        return results

    def _analyze_density(self) -> Dict:
        """Analyze backbone density at different alpha levels."""
        alphas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        pval_offdiag = self._get_tested_pvalues()
        total_tested = len(pval_offdiag)  # edges with p-values

        densities, edge_counts = [], []
        for a in alphas:
            n_sig = np.sum(pval_offdiag <= a)
            densities.append(n_sig / total_tested if total_tested > 0 else 0.0)
            edge_counts.append(n_sig)

        # Count actual backbone edges
        actual_edges = self.backbone.nnz // (1 if self.directed else 2)

        return {
            'alphas': alphas,
            'densities': densities,
            'edge_counts': edge_counts,
            'actual_edges': actual_edges,
            'total_tested': total_tested
        }

    def _analyze_similarity(self) -> Dict:
        """Analyze observed vs expected similarity."""
        if self.obs_sim is None or self.exp_sim is None:
            return {}

        # Get off-diagonal values
        if self.directed:
            mask = ~np.eye(self.n, dtype=bool)
        else:
            mask = np.triu(np.ones((self.n, self.n), dtype=bool), k=1)

        obs = self.obs_sim[mask]
        exp = self.exp_sim[mask]
        pval = self.pvalues[mask]

        # Compute correlations
        pearson = np.corrcoef(obs, exp)[0, 1] if len(obs) > 1 else 0
        spearman = spearmanr(obs, exp)[0] if len(obs) > 1 else 0

        return {
            'obs': obs,
            'exp': exp,
            'pval': pval,
            'pearson': pearson,
            'spearman': spearman
        }

    def _print_summary(self, results: Dict):
        """Print validation summary."""
        pvdist = results['pvalue_dist']
        print(f"  P-value distribution: mean={pvdist['mean']:.4f}, median={pvdist['median']:.4f}\n")

        if results['similarity'] is not None:
            sim = results['similarity']
            print(f"  Similarity correlation: Spearman ρ={sim['spearman']:.4f}\n")

    def plot(self, results: Optional[Dict] = None, save_path: Optional[str] = None, show: bool = True):
        """Create basic diagnostic plots for backbone validation."""
        if results is None:
            results = self._results
        if results is None:
            print("No results to plot. Run validate() first.")
            return

        fig, axes = plt.subplots(1, 2, figsize=PlotConfig.WIDE)

        # 1. Density vs alpha
        dens = results['density']
        axes[0].plot(dens['alphas'], dens['densities'], 'o-', color=PlotConfig.PRIMARY, linewidth=2)
        axes[0].set_xlabel('Alpha threshold', fontweight='bold')
        axes[0].set_ylabel('Share of tested edges retained', fontweight='bold')
        axes[0].set_title('Density vs Alpha', fontweight='bold')
        axes[0].set_xscale('log')
        axes[0].grid(alpha=PlotConfig.GRID_ALPHA)

        # 2. P-value distribution (tested pairs only)
        pval_offdiag = self._get_tested_pvalues()
        mean_pval = np.mean(pval_offdiag)
        var_pval = np.var(pval_offdiag)
        std_pval = np.std(pval_offdiag)

        axes[1].hist(pval_offdiag, bins=50, color=PlotConfig.PRIMARY, alpha=0.7, edgecolor='black')

        # Compute and show corrected threshold using settings from validate_projection
        alpha = self.alpha
        correction = self.correction
        pvalues_array = pval_offdiag
        if len(pvalues_array) > 0 and alpha is not None and correction and correction != 'none':
            m = len(pvalues_array)
            if correction == 'bonferroni':
                corrected_alpha = alpha / m
                axes[1].axvline(corrected_alpha, color='orange', linestyle='-.', linewidth=2,
                                label=f'α_corrected={corrected_alpha:.2e} (Bonferroni)')
            elif correction == 'fdr':
                sorted_pvals = np.sort(pvalues_array)
                threshold_line = alpha * np.arange(1, m + 1) / m
                sig_mask = sorted_pvals <= threshold_line
                if np.any(sig_mask):
                    fdr_threshold = sorted_pvals[np.where(sig_mask)[0][-1]]
                else:
                    fdr_threshold = alpha / m  # fallback: most conservative FDR
                axes[1].axvline(fdr_threshold, color='orange', linestyle='-.', linewidth=2,
                                label=f'α_corrected={fdr_threshold:.2e} (FDR)')

        axes[1].axvline(mean_pval, color='green', linestyle=':', linewidth=2, label=f'μ={mean_pval:.4f}')
        axes[1].set_xlabel('P-value', fontweight='bold')
        axes[1].set_ylabel('Count', fontweight='bold')
        axes[1].set_title(f'P-value Distribution (σ²={var_pval:.6f}, σ={std_pval:.4f})', fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=PlotConfig.GRID_ALPHA)

        _finalize_plot(fig, save_path, show)


# ==========================================
# CONVENIENCE FUNCTION
# ==========================================

def validate_backbone(cmvp, plot: bool = True, verbose: bool = True,
                      save_path: Optional[str] = None, show: bool = True) -> BackboneValidator:
    """
    Validate extracted backbone.

    Parameters
    ----------
    cmvp : CMVP
        Fitted and validated CMVP object
    plot : bool
        Whether to create plots
    verbose : bool
        Whether to print progress messages
    save_path, show
        Forwarded to BackboneValidator.plot() when plot=True.

    Returns
    -------
    BackboneValidator
        Validator object with results
    """
    validator = BackboneValidator(cmvp)
    validator.validate(plot=plot, verbose=verbose, save_path=save_path, show=show)
    return validator

