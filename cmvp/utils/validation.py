"""
CMVP Validation Framework - Lightweight
========================================

Two-part validation framework for Configuration Model Validated Projections.
Focuses on analysis and visualization using matrices from core.py.

Classes:
    PlotConfig: Unified plotting configuration
    BaseValidator: Base class for shared utilities

Functions:
    _finalize_plot: Helper for consistent plot finalization
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional


class PlotConfig:
    """Unified plotting configuration for all validation plots."""

    # Figure sizes — single/fixed-panel layouts
    SMALL = (8, 6)
    MEDIUM = (10, 7)
    LARGE = (12, 8)
    WIDE = (14, 6)
    EXTRA_WIDE = (16, 5)
    SQUARE = (8, 8)

    # Figure sizes — common multi-panel row layouts (width scales with
    # panel count at a fixed ~5.5-6 per-panel height, matching SMALL/WIDE).
    TWO_PANEL = (12, 5.5)
    THREE_PANEL = (18, 5.5)

    # Colors
    PRIMARY = '#2E86AB'
    SECONDARY = '#A23B72'
    ACCENT = '#F18F01'
    SUCCESS = '#06A77D'
    WARNING = '#F77F00'
    DANGER = '#D62828'

    # Style defaults
    DPI = 300
    GRID_ALPHA = 0.3
    SCATTER_ALPHA = 0.5
    LINE_WIDTH = 2.0
    MARKER_SIZE = 5
    FONT_SIZE_TITLE = 13
    FONT_SIZE_LABEL = 12
    FONT_SIZE_LEGEND = 10
    FONT_SIZE_ANNOTATION = 9   # in-axes text boxes (stat callouts, corner labels)
    FONT_SIZE_TICK = 8         # tick labels, dense legends, small in-panel legends
    FONT_SIZE_SUPTITLE = 14    # fig.suptitle() on multi-panel figures — one size above FONT_SIZE_TITLE

    # kwargs for fig.suptitle(...)/plt.suptitle(...) — axes.titlesize doesn't
    # apply to figure-level suptitles, so every call site must pass this
    # explicitly to stay in sync with per-axes titles.
    SUPTITLE_KW = {'fontsize': FONT_SIZE_SUPTITLE, 'fontweight': 'bold'}


# Applied at import time so every plot in the package — even call sites that
# don't pass fontsize=/fontweight= explicitly — renders titles/labels/ticks/
# legends at the same sizes, instead of silently falling back to matplotlib's
# defaults and drifting out of sync with PlotConfig.
plt.rcParams.update({
    'axes.titlesize': PlotConfig.FONT_SIZE_TITLE,
    'axes.titleweight': 'bold',
    'axes.labelsize': PlotConfig.FONT_SIZE_LABEL,
    'axes.labelweight': 'bold',
    'legend.fontsize': PlotConfig.FONT_SIZE_LEGEND,
    'xtick.labelsize': PlotConfig.FONT_SIZE_TICK,
    'ytick.labelsize': PlotConfig.FONT_SIZE_TICK,
})


def ecdf_xy(values: np.ndarray):
    """
    (sorted_values, cdf) step-function coordinates for an empirical CDF:
    cdf[k] = fraction of `values` <= sorted_values[k]. Plot with
    `ax.plot(x, y)` or `ax.step(x, y, where='post')`. Shared by any plot that
    draws an ECDF from a 1-D sample (Fano-factor CDF, p-value CDF comparison).
    Returns two empty arrays if `values` is empty.
    """
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return values, values
    sorted_values = np.sort(values)
    cdf = np.arange(1, n + 1) / n
    return sorted_values, cdf


def sample_adjacency_from_p(P: np.ndarray, is_directed: bool,
                            rng: np.random.Generator) -> np.ndarray:
    """
    Bernoulli-sample one adjacency matrix from an edge-probability matrix.

    Draws each (i, j) pair independently as an edge with probability P[i, j];
    for undirected graphs only the upper triangle is drawn and mirrored, so
    the result is guaranteed symmetric. Shared by any null-model resampling
    routine (bootstrap CIs, calibration checks, backbone null distributions).
    """
    n = P.shape[0]
    A = np.zeros((n, n))
    if is_directed:
        mask = ~np.eye(n, dtype=bool)
        A[mask] = rng.random(mask.sum()) < P[mask]
    else:
        upper = np.triu_indices(n, k=1)
        edges = rng.random(len(upper[0])) < P[upper[0], upper[1]]
        A[upper[0][edges], upper[1][edges]] = 1
        A += A.T
    return A


def _finalize_plot(fig=None, save_path: Optional[str] = None, show: bool = True,
                   dpi: int = PlotConfig.DPI, transparent: bool = True):
    """Finalize plot: tight layout, save (only if save_path is given), and show."""
    target = fig if fig is not None else plt.gcf()
    target.tight_layout()
    if save_path is not None:
        target.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=transparent)
        print(f"Saved plot to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(target)


def get_categorical_color_map(values):
    """
    Build a val -> RGBA color dict for categorical values, shared across all
    plots so the same category (e.g. ground-truth community) always gets the
    same color regardless of which plot function draws it.

    Picks the smallest qualitative tab colormap that still gives every
    category its own distinct color: tab10 (<=10 categories), tab20
    (<=20), tab20+tab20b+tab20c (<=60), else evenly-spaced HSV hues.
    Mirrors the tiering used by
    external_helpers.external_helper_functions.get_color_map() without
    pulling in that module's graph_tool dependency.
    """
    unique_vals = sorted(set(values), key=str)
    n_unique = len(unique_vals)
    if n_unique <= 10:
        colors = plt.get_cmap('tab10').colors[:n_unique]
    elif n_unique <= 20:
        colors = plt.get_cmap('tab20').colors[:n_unique]
    elif n_unique <= 60:
        base = np.vstack([plt.get_cmap('tab20').colors,
                          plt.get_cmap('tab20b').colors,
                          plt.get_cmap('tab20c').colors])
        colors = base[:n_unique]
    else:
        colors = plt.get_cmap('hsv')(np.linspace(0, 1, n_unique, endpoint=False))
    return {v: colors[i] for i, v in enumerate(unique_vals)}


def _configure_scatter_plot(ax, title: str, xlabel: str, ylabel: str):
    """Configure common scatter plot styling."""
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=PlotConfig.FONT_SIZE_LABEL)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=PlotConfig.FONT_SIZE_LABEL)
    ax.set_title(title, fontweight='bold', fontsize=PlotConfig.FONT_SIZE_TITLE)
    ax.legend(fontsize=PlotConfig.FONT_SIZE_LEGEND)
    ax.grid(alpha=PlotConfig.GRID_ALPHA)


class BaseValidator:
    """
    Base validator with shared utilities for all validators.

    Handles:
    - Matrix extraction and caching
    - Off-diagonal value extraction (directed vs undirected)
    - Graph property access
    """

    def __init__(self, cmvp):
        """Initialize with CMVP object."""
        self.cmvp = cmvp
        self.G = cmvp.graph
        self.n = self.G.number_of_nodes()
        self.directed = getattr(cmvp, 'is_directed', False)
        self.weighted = getattr(cmvp, 'is_weighted', False)

        # Get adjacency matrix (cached or compute)
        if hasattr(cmvp, 'A'):
            self.A = cmvp.A
        else:
            self.A = nx.to_numpy_array(self.G, weight='weight' if self.weighted else None)

        # Get p_matrix (probability matrix from null model)
        self._p_matrix = self._extract_p_matrix()

    def _extract_p_matrix(self):
        """Extract and convert p_matrix to dense if needed."""
        if hasattr(self.cmvp, 'p_matrix'):
            p_mat = self.cmvp.p_matrix
            if p_mat is None:
                return None
            if hasattr(p_mat, 'toarray'):
                return p_mat.toarray()
            return np.array(p_mat)
        return None

    @property
    def p_matrix(self):
        """Get cached dense p_matrix."""
        return self._p_matrix

    def _extract_offdiag_values(self, matrix):
        """Extract off-diagonal values for current graph type."""
        if self.directed:
            mask = ~np.eye(self.n, dtype=bool)
            return matrix[mask]
        else:
            return matrix[np.triu_indices(self.n, k=1)]


__all__ = [
    'PlotConfig',
    'BaseValidator',
    'ecdf_xy',
    'sample_adjacency_from_p',
    'get_categorical_color_map',
    '_finalize_plot',
    '_configure_scatter_plot',
]
