"""
CMVP Backbone Validation - Similarity Measure Comparison Plots
==================================================================

_SimilarityPlotsMixin, mixed into BackboneValidator (see backbone_validation.py).
Plots comparing CMVP's backbone/statistics against alternative similarity
measures and the raw network topology (heatmaps, dendrograms, network layout).
"""

import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from scipy.stats import spearmanr, pearsonr
from typing import Dict, Optional

from .validation import PlotConfig, _finalize_plot, get_categorical_color_map
from ..core import directed_matmul


class _SimilarityPlotsMixin:
    """Similarity-measure comparison plots for BackboneValidator."""

    def plot_dendrogram(self, matrix: np.ndarray = None, label: str = None,
                        method: str = 'average',
                        distance: str = 'profile',
                        color_attr: Optional[str] = None,
                        node_label_attr: Optional[str] = None,
                        show_labels: Optional[bool] = None, save_path: Optional[str] = None,
                        show: bool = True):
        """
        Clustered heatmap from a node-by-node matrix.

        Parameters
        ----------
        matrix : np.ndarray, optional
            Matrix to plot. If None, uses cmvp.test_sim_matrix_sig.
        label : str, optional
            Label for the colorbar. If None, auto-detected from transform
            (only when using default test_sim_matrix_sig).
        method : str
            Linkage method ('average', 'complete', 'single', 'weighted').
        distance : str
            How to derive pairwise distances for clustering (default: 'profile').
            'profile': euclidean distance between matrix rows (similar profiles cluster).
            'affinity': treat entries as similarities, cluster on max - value
            (high/positive entries = close, negative = far). The heatmap still
            shows the original values. Use with method='average' or 'complete'.
        color_attr : str, optional
            Node attribute name to color tick labels by (e.g. ground-truth community).
        node_label_attr : str, optional
            Node attribute name to display as the tick label text instead of the
            raw node id (e.g. a human-readable node name). Falls back to the
            node id for nodes missing the attribute.
        show_labels : bool, optional
            Show node labels on axes. If None (default), auto-hidden for n > 50.
            If explicitly True or False, that always wins regardless of node count.
        save_path : str, optional
            Path to save figure
        show : bool
            Display the plot (default True)
        """
        import seaborn as sns
        import pandas as pd
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import pdist

        # Track whether caller passed their own matrix
        using_default = (matrix is None)

        if matrix is None:
            matrix = self.cmvp.test_sim_matrix_sig
            if matrix is None:
                raise ValueError("No matrix provided and test_sim_matrix_sig not found. Call validate_projection() first.")
        matrix = matrix.toarray() if hasattr(matrix, 'toarray') else matrix

        # Auto-detect label only when using the default test_sim_matrix_sig
        transform = self.cmvp._config.get('transform', 'pvalue')
        if label is not None:
            cbar_label = label
        elif using_default:
            cbar_label = {'pvalue': 'p-value', 'neglog': '-log(p)', 'logneglog': 'log(-log(p))'}.get(transform, transform)
        else:
            cbar_label = 'Similarity'

        # Work on a copy
        matrix = np.array(matrix, dtype=float)

        # --- Symmetry from directed_mode ---
        directed_mode_used = self.cmvp._config.get('directed_mode', 'out-out')
        is_symmetric = (not self.G.is_directed()) or directed_mode_used in ('out-out', 'in-in')

        # --- DataFrame with node labels ---
        nodes = list(self.G.nodes())
        n_nodes = len(nodes)
        df = pd.DataFrame(matrix, index=nodes, columns=nodes)

        # --- Compute linkage ---
        from scipy.spatial.distance import squareform
        # Only treat as raw p-values when using default matrix with pvalue transform
        is_pvalue = (using_default and transform == 'pvalue')
        matrix_dense = matrix.toarray() if hasattr(matrix, 'toarray') else matrix
        if distance not in ('profile', 'affinity'):
            raise ValueError(f"distance must be 'profile' or 'affinity', got '{distance}'")
        if distance == 'affinity':
            # Entries are similarities: flip monotonically so high/positive = close
            dist_matrix = matrix_dense.max() - matrix_dense
            np.fill_diagonal(dist_matrix, 0)
            condensed = squareform(dist_matrix, checks=False)
            row_link = linkage(condensed, method=method)
        elif is_pvalue:
            dist_matrix = matrix_dense.copy()
            np.fill_diagonal(dist_matrix, 0)
            condensed = squareform(dist_matrix, checks=False)
            row_link = linkage(condensed, method=method)
        else:
            condensed = pdist(matrix_dense, metric='euclidean')
            row_link = linkage(condensed, method=method)

        # --- 5. Build color map from node attribute (before plotting) ---
        node_colors = None
        val_to_color = None
        color_series = None
        if color_attr is not None:
            attr_values = nx.get_node_attributes(self.G, color_attr)
            if attr_values:
                val_to_color = get_categorical_color_map(attr_values.values())
                node_colors = {node: val_to_color.get(attr_values.get(node), (0.5, 0.5, 0.5, 1.0))
                               for node in nodes}
                # Color strip for row_colors / col_colors
                color_series = pd.Series(
                    [val_to_color.get(attr_values.get(n), (0.5, 0.5, 0.5, 1.0)) for n in nodes],
                    index=nodes, name=color_attr)

        # --- 6. Plot ---
        fig_size = min(20, max(10, n_nodes * 0.3))
        # Diverging data (e.g. z-scores, signed backbones): coolwarm centered at 0
        has_both_signs = (matrix_dense.min() < 0) and (matrix_dense.max() > 0)
        if has_both_signs:
            cmap = "coolwarm"
        elif is_pvalue:
            cmap = "YlOrRd_r"
        else:
            cmap = "YlOrRd"
        clustermap_kw = dict(
            cmap=cmap,
            center=0 if has_both_signs else None,
            figsize=(fig_size, fig_size),
            cbar_kws={"label": cbar_label, "fraction": 0.046, "shrink": 0.6},
            dendrogram_ratio=(0.12, 0.12),
            linewidths=0,
            tree_kws={"linewidths": 1.5},
        )
        if color_series is not None:
            clustermap_kw['row_colors'] = color_series
            clustermap_kw['col_colors'] = color_series

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, 10 * n_nodes))
        try:
            if is_symmetric:
                g = sns.clustermap(df, row_linkage=row_link, col_linkage=row_link,
                                   **clustermap_kw)
            else:
                if distance == 'affinity':
                    col_dist = matrix_dense.T.max() - matrix_dense.T
                    np.fill_diagonal(col_dist, 0)
                    col_condensed = squareform(col_dist, checks=False)
                elif is_pvalue:
                    col_condensed = squareform(matrix_dense.T.copy(), checks=False)
                else:
                    col_condensed = pdist(matrix_dense.T, metric='euclidean')
                col_link = linkage(col_condensed, method=method)
                g = sns.clustermap(df, row_linkage=row_link, col_linkage=col_link,
                                   **clustermap_kw)
        finally:
            sys.setrecursionlimit(old_limit)

        # --- 7. Style dendrograms ---
        for ax in (g.ax_row_dendrogram, g.ax_col_dendrogram):
            ax.set_facecolor('white')
            for spine in ax.spines.values():
                spine.set_visible(False)

        # --- 9. Title and labels ---
        if distance == 'affinity':
            dist_label = 'affinity (max - value)'
        elif is_pvalue:
            dist_label = 'pvalue (raw)'
        else:
            dist_label = 'euclidean'
        g.fig.suptitle(
            f'Clustered Heatmap  ({cbar_label})\ndistance={dist_label}  |  linkage={method}',
            y=1.02, **PlotConfig.SUPTITLE_KW)
        g.ax_heatmap.set_xlabel('Node', fontweight='bold', fontsize=PlotConfig.FONT_SIZE_LABEL)
        g.ax_heatmap.set_ylabel('Node', fontweight='bold', fontsize=PlotConfig.FONT_SIZE_LABEL)

        effective_show_labels = (n_nodes <= 50) if show_labels is None else show_labels
        if not effective_show_labels:
            g.ax_heatmap.set_xticklabels([])
            g.ax_heatmap.set_yticklabels([])
        else:
            g.ax_heatmap.set_xticklabels(
                g.ax_heatmap.get_xticklabels(), rotation=90, ha='center', fontsize=PlotConfig.FONT_SIZE_TICK)
            g.ax_heatmap.set_yticklabels(
                g.ax_heatmap.get_yticklabels(), rotation=0, ha='left', va='center', fontsize=PlotConfig.FONT_SIZE_TICK)

            if node_colors is not None:
                for label in g.ax_heatmap.get_xticklabels():
                    node = label.get_text()
                    if node in node_colors:
                        label.set_color(node_colors[node])
                        label.set_fontweight('bold')
                for label in g.ax_heatmap.get_yticklabels():
                    node = label.get_text()
                    if node in node_colors:
                        label.set_color(node_colors[node])
                        label.set_fontweight('bold')

            if node_label_attr is not None:
                attr_values = nx.get_node_attributes(self.G, node_label_attr)
                label_map = {str(n): str(attr_values.get(n, n)) for n in nodes}
                for label in g.ax_heatmap.get_xticklabels():
                    label.set_text(label_map.get(label.get_text(), label.get_text()))
                for label in g.ax_heatmap.get_yticklabels():
                    label.set_text(label_map.get(label.get_text(), label.get_text()))

        # --- 10. Legend for color_attr ---
        if val_to_color is not None and color_attr is not None:
            from matplotlib.patches import Patch
            legend_handles = [Patch(facecolor=c, label=str(v))
                              for v, c in val_to_color.items()]
            g.ax_heatmap.legend(handles=legend_handles, title=color_attr,
                                loc='upper left', bbox_to_anchor=(1.05, 1),
                                fontsize=PlotConfig.FONT_SIZE_TICK, title_fontsize=PlotConfig.FONT_SIZE_TICK)

        g.fig.subplots_adjust(top=0.95)
        if save_path:
            g.savefig(save_path, dpi=PlotConfig.DPI, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_similarity_heatmap(self, similarity: str = 'common_neighbors',
                               directed_mode: Optional[str] = None,
                               save_path: Optional[str] = None, show: bool = True):
        """
        Create 2x2 panel heatmap showing expected, observed, validated, and backbone.

        Visualizes the relationship between observed and expected similarities from the
        null model, highlighting areas where the backbone deviates from expectations.

        Parameters
        ----------
        similarity : str
            Similarity measure to compute ('common_neighbors', 'jaccard', 'salton', etc.)
        directed_mode : str
            For directed graphs, which overlap to use:
            - 'out-out': both nodes' out-neighbors
            - 'in-in': both nodes' in-neighbors
            - 'out-in': u's out-neighbors vs v's in-neighbors
            - 'in-out': u's in-neighbors vs v's out-neighbors
        save_path : str, optional
            Path to save the figure
        show : bool
            Whether to display the plot (default: True)
        """
        import seaborn as sns
        from scipy import sparse as sp

        # Resolve directed_mode: prefer explicit arg, fall back to what was used in validate_projection
        if directed_mode is None:
            directed_mode = (self.cmvp._config.get('directed_mode') or 'out-out')

        nodes = list(self.G.nodes())
        n = len(nodes)

        # Use stored matrices from CMVP when available
        if self.obs_sim is not None:
            observed_sim = self.obs_sim
        else:
            observed_sim = self.cmvp.compute_similarity_matrix(
                measure=similarity, sparse_output=False, directed_mode=directed_mode)

        if self.exp_sim is not None:
            expected_sim = self.exp_sim
        else:
            # Analytical fallback (direction-aware)
            P = self.p_matrix
            is_directed = self.cmvp.is_directed

            expected_cn, deg_u, deg_v = directed_matmul(P, directed_mode, is_directed)
            np.fill_diagonal(expected_cn, 0)
            if similarity == 'common_neighbors':
                expected_sim = expected_cn
            elif similarity == 'jaccard':
                expected_union = deg_u[:, np.newaxis] + deg_v[np.newaxis, :] - expected_cn
                expected_sim = np.divide(expected_cn, expected_union,
                                        out=np.zeros_like(expected_cn), where=expected_union > 0)
            else:
                expected_sim = expected_cn

        # Convert sparse to dense if needed
        if sp.issparse(observed_sim):
            observed_sim = observed_sim.toarray()
        if sp.issparse(expected_sim):
            expected_sim = expected_sim.toarray()

        # Zero the diagonals for display
        np.fill_diagonal(observed_sim, 0)
        np.fill_diagonal(expected_sim, 0)

        # Get validated similarity matrix (uses transform from validate_projection)
        _vsig = self.cmvp.test_sim_matrix_sig
        validated_matrix = _vsig.toarray() if hasattr(_vsig, 'toarray') else _vsig.copy()
        np.fill_diagonal(validated_matrix, 0)
        transform = self.cmvp._config.get('transform', 'pvalue')
        val_label = {'pvalue': 'p-value', 'neglog': '-log(p)', 'logneglog': 'log(-log(p))'}.get(transform, transform)
        val_cmap = 'YlOrRd_r' if transform == 'pvalue' else 'YlOrRd'

        # Shared scale for observed vs expected (same units)
        cn_vmax = max(observed_sim.max(), expected_sim.max())
        if cn_vmax == 0:
            cn_vmax = 1

        sim_label = similarity.replace('_', ' ').title()

        # Backbone binary mask (use cmvp.backbone which reflects filter_backbone())
        filtered_bb = self.cmvp.backbone
        if sp.issparse(filtered_bb):
            filtered_bb = filtered_bb.toarray()
        backbone_binary = (filtered_bb > 0).astype(float)
        np.fill_diagonal(backbone_binary, 0)

        # Create 2×2 plot grid
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 12.5))
        sns.set_style("white")

        title_size = PlotConfig.FONT_SIZE_TITLE + 9
        cbar_label_size = PlotConfig.FONT_SIZE_LABEL + 5
        cbar_tick_size = PlotConfig.FONT_SIZE_TICK + 5

        # === Row 1: Expected, Observed ===
        hm1 = sns.heatmap(
            expected_sim, ax=axes[0, 0], cmap="Greens",
            vmin=0, vmax=cn_vmax, cbar_kws={"fraction": 0.046}
        )
        hm1.collections[0].colorbar.ax.tick_params(labelsize=cbar_tick_size)
        axes[0, 0].set_title(f"Expected {sim_label}", size=title_size, fontweight='bold')

        hm2 = sns.heatmap(
            observed_sim, ax=axes[0, 1], cmap="Blues",
            vmin=0, vmax=cn_vmax, cbar_kws={"fraction": 0.046}
        )
        hm2.collections[0].colorbar.ax.tick_params(labelsize=cbar_tick_size)
        axes[0, 1].set_title(f"Observed {sim_label}", size=title_size, fontweight='bold')

        # === Row 2: Validated, Backbone ===
        hm3 = sns.heatmap(
            validated_matrix, ax=axes[1, 0], cmap=val_cmap,
            cbar_kws={"label": val_label, "fraction": 0.046}
        )
        hm3_cbar = hm3.collections[0].colorbar
        hm3_cbar.ax.tick_params(labelsize=cbar_tick_size)
        hm3_cbar.set_label(val_label, fontsize=cbar_label_size)
        axes[1, 0].set_title(f"Validated Similarity\n({val_label})", size=title_size, fontweight='bold')

        sns.heatmap(
            backbone_binary, ax=axes[1, 1], cmap="Greys",
            vmin=0, vmax=1, cbar=False
        )
        axes[1, 1].set_title(f"Backbone Edges", size=title_size, fontweight='bold')

        # Clean axes
        for row in axes:
            for ax in row:
                ax.set_xticks([])
                ax.set_yticks([])

        _finalize_plot(fig, save_path, show)

    def plot_degree_vs_similarity(self, weighted: bool = False, masked: bool = True,
                                  color_by: str = 'zscore',
                                  save_path: Optional[str] = None, show: bool = True):
        """
        Plot observed vs expected common neighbor similarity as a function of node degree or strength.

        Each dot is one node: x = degree (or strength if weighted=True),
        y = mean observed CN overlap across all its pairs. Marker size is the number
        of CN pairs the node's mean was averaged over (bigger = more reliable). The
        tomato line shows the CM null expectation; the green dotted line the ER
        baseline.

        For weighted=True, strength is a near-continuous sum of edge weights, so
        binning the expected-CN line by exact strength value (as done for integer
        degree) leaves most bins with only 1-2 nodes — a noisy, choppy line rather
        than an aggregate trend. The line is instead built from quantile bins (equal
        node count per bin) when weighted=True, which keeps a real sample size per
        bin regardless of how continuous strength is.

        Parameters
        ----------
        weighted : bool
            If True, use node strength on x-axis (for weighted validate_projection runs).
        color_by : str
            'zscore' (default) — per-node mean (obs - exp) / std (significance).
            'effect' — per-node mean theta-hat = (obs - exp) / var (effect size,
            degree/opportunity term divided out).
            'degree' — structural (unweighted) node degree, useful with
            weighted=True to see whether strength alone explains a node's
            position or whether degree is doing separate work (e.g. same
            strength via many small-weight edges vs. few large-weight ones).
        """
        fig, ax = plt.subplots(figsize=PlotConfig.WIDE)
        self._draw_degree_similarity_panel(ax, weighted=weighted, masked=masked, color_by=color_by)
        _finalize_plot(fig, save_path, show)

    def _draw_degree_similarity_panel(self, ax, weighted: bool = False, masked: bool = True,
                                      color_by: str = 'zscore'):
        """Draws the degree/strength-vs-similarity scatter (see
        plot_degree_vs_similarity's docstring) onto an existing axes — shared
        by that standalone plot and the combined plot_topology_vs_degree
        panel view so both stay visually and numerically identical."""
        if color_by not in ('zscore', 'effect', 'degree'):
            raise ValueError(f"color_by must be 'zscore', 'effect', or 'degree', got '{color_by}'")
        obs = self.cmvp.obs_sim_matrix
        exp = self.cmvp.exp_sim_matrix
        exp_std = self.cmvp.exp_sim_matrix_std

        if obs is None or exp is None:
            raise RuntimeError("Call validate_projection() first to compute similarity matrices.")

        G = self.cmvp.graph
        if weighted:
            weight_attr = self.cmvp._config.get('weight_attr', 'weight')
            if self.cmvp.is_directed:
                node_strength = np.array([sum(d.get(weight_attr, 1) for _, _, d in G.out_edges(n, data=True))
                                          for n in self.cmvp.nodes])
            else:
                node_strength = np.array([sum(d.get(weight_attr, 1) for _, _, d in G.edges(n, data=True))
                                          for n in self.cmvp.nodes])
            deg = node_strength
            x_label = 'Node Strength'
            y_label = 'Mean Weighted Common Neighbor Overlap'
            title = 'Observed vs Expected Weighted Similarity by Strength'
        else:
            if self.cmvp.is_directed:
                deg = np.array([d for _, d in G.out_degree(self.cmvp.nodes)])
            else:
                deg = np.array([d for _, d in G.degree(self.cmvp.nodes)])
            x_label = 'Node Degree'
            y_label = 'Mean Common Neighbor Overlap'
            title = 'Observed vs Expected Similarity by Degree'

        n = self.cmvp.n

        # Structural (unweighted) degree — always computed, used for the
        # 'degree' color_by option even when weighted=True uses strength on
        # the x-axis instead.
        if self.cmvp.is_directed:
            struct_deg = np.array([d for _, d in G.out_degree(self.cmvp.nodes)])
        else:
            struct_deg = np.array([d for _, d in G.degree(self.cmvp.nodes)])

        idx = np.arange(n)
        if masked:
            def _mean_node(i, obs_row, ref_row):
                mask = (idx != i) & (obs_row > 0)
                return ref_row[mask].mean() if mask.any() else np.nan

            def _n_pairs(i, obs_row):
                return int(((idx != i) & (obs_row > 0)).sum())
        else:
            def _mean_node(i, obs_row, ref_row):
                mask = idx != i
                return ref_row[mask].mean()

            def _n_pairs(i, obs_row):
                return int(n - 1)

        node_obs = np.array([_mean_node(i, obs[i], obs[i]) for i in range(n)])
        node_exp = np.array([_mean_node(i, obs[i], exp[i]) for i in range(n)])
        node_std = np.array([_mean_node(i, obs[i], exp_std[i]) for i in range(n)]) \
                   if exp_std is not None else np.zeros(n)
        node_n_pairs = np.array([_n_pairs(i, obs[i]) for i in range(n)])

        # Drop nans (only possible when masked=True for nodes with no shared neighbors)
        valid = ~np.isnan(node_obs) & ~np.isnan(node_exp)
        sort_idx = np.argsort(deg[valid])
        sorted_deg = deg[valid][sort_idx]
        sorted_obs = node_obs[valid][sort_idx]
        sorted_exp = node_exp[valid][sort_idx]
        sorted_std = node_std[valid][sort_idx]
        sorted_n_pairs = node_n_pairs[valid][sort_idx]
        sorted_struct_deg = struct_deg[valid][sort_idx]

        # Per-node mean z-score or effect size, averaged over off-diagonal pairs
        _std = self.cmvp.exp_sim_matrix_std
        sorted_sig = None
        cmap = 'coolwarm'
        if color_by == 'degree':
            sorted_sig = sorted_struct_deg.astype(float)
            sig_label = 'Node degree'
            cmap = 'viridis'
        elif obs is not None and exp is not None and _std is not None:
            with np.errstate(invalid='ignore', divide='ignore'):
                if color_by == 'zscore':
                    stat_matrix = np.where(_std > 0, (obs - exp) / _std, np.nan)
                    sig_label = 'Mean z-score'
                else:
                    _var = _std ** 2
                    stat_matrix = np.where(_var > 0, (obs - exp) / _var, np.nan)
                    sig_label = r'Mean $\widehat{\mathrm{Effect}}$'
            np.fill_diagonal(stat_matrix, np.nan)
            node_sig = np.array([
                np.nanmean(stat_matrix[i, np.arange(n) != i]) for i in range(n)
            ])
            sorted_sig = node_sig[valid][sort_idx]

        # Marker size ~ sample size behind each node's mean (bigger = more
        # CN pairs it was averaged over, i.e. a more reliable point).
        max_pairs = max(sorted_n_pairs.max(), 1)
        marker_sizes = 10 + 60 * np.sqrt(sorted_n_pairs / max_pairs)

        if weighted:
            # Strength is near-continuous, so exact-value grouping (fine for
            # integer degree) leaves ~1 node per bin. Use quantile bins
            # (equal node count per bin) instead, so the line reflects an
            # actual local average rather than per-node noise.
            n_bins = min(25, max(5, len(sorted_deg) // 5))
            edges = np.unique(np.quantile(sorted_deg, np.linspace(0, 1, n_bins + 1)))
            if len(edges) < 2:
                edges = np.array([sorted_deg.min(), sorted_deg.max() + 1e-9])
            bin_id = np.clip(np.searchsorted(edges, sorted_deg, side='right') - 1,
                             0, len(edges) - 2)
            unique_degs, bin_exp_mean, bin_exp_std = [], [], []
            for b in range(len(edges) - 1):
                bm = bin_id == b
                if bm.any():
                    unique_degs.append(sorted_deg[bm].mean())
                    bin_exp_mean.append(sorted_exp[bm].mean())
                    bin_exp_std.append(sorted_std[bm].mean())
            unique_degs = np.array(unique_degs)
            bin_exp_mean = np.array(bin_exp_mean)
            bin_exp_std = np.array(bin_exp_std)
        else:
            unique_degs = np.unique(sorted_deg)
            bin_exp_mean = np.array([sorted_exp[sorted_deg == d].mean() for d in unique_degs])
            bin_exp_std  = np.array([sorted_std[sorted_deg == d].mean() for d in unique_degs])

        if sorted_sig is not None:
            if color_by == 'degree':
                vmin, vmax = np.nanmin(sorted_sig), np.nanmax(sorted_sig)
            else:
                z_abs = np.nanmax(np.abs(sorted_sig))
                vmin, vmax = -z_abs, z_abs
            sc = ax.scatter(sorted_deg, sorted_obs, c=sorted_sig, cmap=cmap,
                            vmin=vmin, vmax=vmax, s=marker_sizes,
                            alpha=0.8, linewidth=0, zorder=2, rasterized=True)
            plt.colorbar(sc, ax=ax, label=sig_label)
        else:
            ax.scatter(sorted_deg, sorted_obs, s=marker_sizes, alpha=0.6, color='steelblue',
                       label='Observed (per-node mean)', zorder=2)
        ax.plot(unique_degs, bin_exp_mean, color='tomato', linewidth=2,
                label='Expected CN (mean)', zorder=3)
        ax.fill_between(unique_degs, bin_exp_mean - bin_exp_std, bin_exp_mean + bin_exp_std,
                        color='tomato', alpha=0.2, label='Expected (±std)', zorder=2)

        if not weighted:
            m = self.cmvp.graph.number_of_edges()
            p_er = m / (n * (n - 1) / 2) if not self.cmvp.is_directed else m / (n * (n - 1))
            er_baseline = (n - 2) * p_er ** 2
            ax.axhline(er_baseline, color='seagreen', linewidth=2, linestyle=':',
                       label=f'Erdős-Rényi baseline ({er_baseline:.2f})', zorder=3)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        x_max = sorted_deg.max()
        if weighted:
            ticks = np.linspace(0, x_max, 11)
            ax.set_xticks(np.round(ticks, 1))
        else:
            step = max(1, round(x_max / 10 / 10) * 10) if x_max >= 10 else 1
            ax.set_xticks(np.arange(0, x_max + step, step))
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend()

    def plot_observed_vs_expected_cn(self, mask_zero_obs: bool = True,
                                      save_path: Optional[str] = None, show: bool = True):
        """
        Observed CN (colored by z-score) against degree product, with E[CN] ± std
        shown as a grey line and band.

        Each point is one node pair (i, j). Also overlays the Chung-Lu (configuration
        model, first-order) approximation E[CN]_ij ~= k_i*k_j*sum(k_k^2) / (2m)^2 as a
        reference line — a degree-only estimate that skips CMVP's null-model fit
        entirely, useful for seeing how much the exact fit buys over the naive
        degree-product approximation.

        Parameters
        ----------
        mask_zero_obs : bool
            If True (default), only show pairs where observed CN > 0.
        """
        obs = self.cmvp.obs_sim_matrix
        exp = self.cmvp.exp_sim_matrix
        exp_std = self.cmvp.exp_sim_matrix_std

        if obs is None or exp is None:
            raise RuntimeError("Call validate_projection() first to compute similarity matrices.")

        n = self.cmvp.n
        triu = np.triu_indices(n, k=1)
        obs_vals = obs[triu]
        exp_vals = exp[triu]

        if mask_zero_obs:
            mask = obs_vals > 0
        else:
            mask = np.ones(len(obs_vals), dtype=bool)
        obs_vals = obs_vals[mask]
        exp_vals = exp_vals[mask]

        std_vals = exp_std[triu][mask]
        with np.errstate(invalid='ignore', divide='ignore'):
            z_vals = np.where(std_vals > 0, (obs_vals - exp_vals) / std_vals, 0.0)
        vlim = np.nanpercentile(np.abs(z_vals), 99)

        # Chung-Lu approximation: p_ik ~= k_i*k_k / (2m), so
        # E[CN]_ij ~= sum_k p_ik*p_jk = k_i*k_j * sum_k(k_k^2) / (2m)^2.
        # Degree-only, no null-model fitting — a cheap baseline to compare
        # against CMVP's exact-fit E[CN].
        deg = np.asarray(self.cmvp.A_bin.sum(axis=1)).flatten().astype(float)
        two_m = deg.sum()
        sum_k2 = np.sum(deg ** 2)
        chung_lu_vals = (deg[triu[0]] * deg[triu[1]]) * sum_k2 / two_m ** 2
        chung_lu_vals = chung_lu_vals[mask]

        lim = max(obs_vals.max(), exp_vals.max()) if len(obs_vals) and len(exp_vals) else 1.0
        # Adaptive "nice" tick step (1/2/5 x 10^k) instead of a fixed step of
        # 5 — a fixed step is too coarse for small networks (lim ~ 5-20) and
        # produces thousands of tick labels for large ones (lim ~ 10k+, e.g.
        # airports).
        ticks = MaxNLocator(nbins=10, integer=True, min_n_ticks=3).tick_values(0, lim)
        ticks = ticks[ticks >= 0].astype(int)
        lim_ticked = int(ticks[-1]) if len(ticks) else int(np.ceil(lim))

        def _draw_panel(ax, x_scores, groups, scatter_x, scatter_y, xlabel, ylabel, title):
            sc = ax.scatter(scatter_x, scatter_y, s=12, alpha=0.35, c=z_vals,
                            cmap='RdBu_r', vmin=-vlim, vmax=vlim, zorder=2)
            fig.colorbar(sc, ax=ax, label='z-score  (obs − exp) / std', pad=0.12)

            ax.boxplot(groups, positions=x_scores, widths=0.8,
                      patch_artist=True, showfliers=False, zorder=3,
                      boxprops=dict(facecolor='lightgrey', color='grey', alpha=0.5),
                      medianprops=dict(color='black'),
                      whiskerprops=dict(color='grey'),
                      capprops=dict(color='grey'))

            ax.plot([0, lim_ticked], [0, lim_ticked], 'k--', lw=1, label='y = x')

            pear_r, _ = pearsonr(scatter_x, scatter_y)
            spear_r, _ = spearmanr(scatter_x, scatter_y)
            ax.text(0.98, 0.02, f'Pearson r = {pear_r:.3f}\nSpearman ρ = {spear_r:.3f}',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=PlotConfig.FONT_SIZE_ANNOTATION,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks.astype(int))
            ax.set_yticks(ticks)
            # AutoMinorLocator subdivides relative to the major tick step, so
            # with the adaptive "nice" major ticks above (always a bounded
            # count) it stays well under MAXTICKS at any network scale —
            # unlike a fixed MultipleLocator(1)/small step, which used to
            # blow up on graphs with large CN counts (e.g. airports).
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='both', which='minor', length=3)
            ax.set_xlim(-lim_ticked * 0.05, lim_ticked + 0.5)
            ax.set_ylim(0, lim_ticked + 0.5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(fontsize=PlotConfig.FONT_SIZE_TICK)
            ax.grid(alpha=PlotConfig.GRID_ALPHA)

        fig, ax_obs = plt.subplots(figsize=PlotConfig.SMALL)

        # x = observed CN (dense integer range so empty scores still get a
        # step), boxplot shows E[CN] distribution per observed score.
        cn_scores = np.arange(int(obs_vals.min()), int(obs_vals.max()) + 1)
        exp_by_score = [exp_vals[obs_vals == s] for s in cn_scores]
        _draw_panel(ax_obs, cn_scores, exp_by_score, obs_vals, exp_vals,
                   'Observed Common Neighbors', 'E[CN]',
                   'E[CN] Distribution per Observed CN Score')

        # Chung-Lu reference line: mean degree-only approximation per observed
        # CN score, overlaid on top of the CMVP E[CN] boxplots above.
        cl_by_score = [chung_lu_vals[obs_vals == s] for s in cn_scores]
        cl_means = np.array([g.mean() if len(g) else np.nan for g in cl_by_score])
        ax_obs.plot(cn_scores, cl_means, color='tab:orange', lw=1.5, marker='.',
                   markersize=4, alpha=0.9, label='Chung-Lu approx (mean)', zorder=4)
        ax_obs.legend(fontsize=PlotConfig.FONT_SIZE_TICK)

        _finalize_plot(fig, save_path, show)

    def plot_topology_vs_degree(self, weighted: bool = False, masked: bool = True,
                                color_by: str = 'zscore',
                                save_path: Optional[str] = None, show: bool = True):
        """
        Three-panel degree-diagnostic figure: degree/strength-vs-similarity
        (see plot_degree_vs_similarity), then ANND and local clustering vs
        degree, each comparing the observed network against the CM null
        model expectation. weighted/masked/color_by are forwarded to the
        similarity panel only — see plot_degree_vs_similarity's docstring.

        For each node i:
          - Expected ANND:       E[knn_i] = Σ_j p_ij * k_j / k_i
          - Expected clustering: E[c_i]   = Σ_{j≠k} p_ij * p_jk * p_ki / (k_i*(k_i-1))
        """
        G = self.cmvp.graph
        nodes = self.cmvp.nodes
        n = self.cmvp.n
        P = self.cmvp.p_matrix  # (n, n) connection probability matrix

        deg = np.array([G.degree(v) for v in nodes], dtype=float)
        degrees = np.array([G.degree(v) for v in nodes], dtype=float)

        # Observed ANND and clustering
        obs_annd = np.array([
            np.mean([G.degree(nb) for nb in G.neighbors(v)]) if G.degree(v) > 0 else np.nan
            for v in nodes
        ])
        obs_clustering = np.array([nx.clustering(G, v) for v in nodes])

        # Expected ANND under CM: E[knn_i] = (P @ degrees) / k_i
        exp_annd = np.where(deg > 0, (P @ degrees) / deg, np.nan)
        # Analytical std: Var[knn_i] = Σ_j k_j² * p_ij*(1-p_ij) / k_i²
        std_annd = np.where(
            deg > 0,
            np.sqrt((P * (1 - P)) @ (degrees ** 2)) / deg,
            np.nan
        )

        # Expected clustering under CM: E[c_i] = Σ_{j,k} p_ij*p_jk*p_ki / (k_i*(k_i-1))
        P_sq = P @ P
        tri_closed = np.array([P_sq[i] @ P[i] for i in range(n)])  # (P^3)_ii
        denom = deg * (deg - 1)
        exp_clustering = np.where(denom > 0, np.clip(tri_closed / denom, 0, 1), np.nan)

        # Analytical Var[T_i]: triangles sharing edges are correlated.
        # Var[T_i] = Σ_{j,k} q_ijk*(1-q_ijk) + 2*Σ_{j,k<l} p_ij*(p_jk*p_ki*p_jl*p_li - q_ijk*q_ijl)
        # where q_ijk = p_ij*p_jk*p_ki
        std_clustering = np.full(n, np.nan)
        for i in range(n):
            if denom[i] <= 0:
                continue
            pi = P[i]           # (n,) connection probs from i
            # q[j,k] = p_ij * p_jk * p_ki  — probability of triangle i-j-k
            Q = np.outer(pi, pi) * P   # Q[j,k] = p_ij * p_jk * p_ki
            np.fill_diagonal(Q, 0)
            Q[i, :] = 0
            Q[:, i] = 0
            # Variance of triangle count T_i
            var_T = np.sum(Q * (1 - Q))
            # Covariance from pairs of triangles sharing edge (i,j):
            # Cov((i,j,k),(i,j,l)) = p_ij*(p_jk*p_ki*p_jl*p_li) - q_ijk*q_ijl
            # Summed over j, and all k<l pairs: = Σ_j p_ij * (Σ_k p_jk*p_ki)^2 - (Σ_k q_ijk)^2
            for j in range(n):
                if j == i or pi[j] == 0:
                    continue
                col = P[j] * pi          # p_jk * p_ki for all k
                col[i] = 0
                col[j] = 0
                sum_col = col.sum()      # Σ_k p_jk*p_ki
                sum_q_sq = (pi[j] * col).sum()  # Σ_k q_ijk (= p_ij * Σ_k p_jk*p_ki)
                var_T += 2 * (pi[j] * sum_col ** 2 - sum_q_sq ** 2)
            std_clustering[i] = np.sqrt(max(var_T, 0)) / denom[i]

        fig, axes = plt.subplots(1, 3, figsize=PlotConfig.THREE_PANEL,
                                 gridspec_kw={'width_ratios': [1.2, 1, 1]})

        self._draw_degree_similarity_panel(axes[0], weighted=weighted, masked=masked, color_by=color_by)

        m = G.number_of_edges()
        mean_deg = 2 * m / n
        p_er = m / (n * (n - 1) / 2)
        er_baselines = [mean_deg, p_er]

        for ax, obs, exp, exp_std_arr, std_label, y_label, title, er_val, y_max in [
            (axes[1], obs_annd, exp_annd, std_annd,
             'Expected ±1σ (analytical)',
             'Average Nearest-Neighbor Degree', 'ANND vs Degree', er_baselines[0], None),
            (axes[2], obs_clustering, exp_clustering, std_clustering,
             'Expected ±1σ (analytical)',
             'Local Clustering Coefficient', 'Clustering vs Degree', er_baselines[1], 1.1),
        ]:
            valid = ~np.isnan(obs) & ~np.isnan(exp)
            ax.scatter(deg[valid], obs[valid], s=25, alpha=0.6,
                       color='steelblue', label='Observed', zorder=2)

            # Bin expected by degree for a clean line
            unique_degs = np.unique(deg[valid])
            bin_exp = np.array([exp[valid][deg[valid] == d].mean() for d in unique_degs])
            bin_std = np.array([exp_std_arr[valid][deg[valid] == d].mean() for d in unique_degs])

            ax.plot(unique_degs, bin_exp, color='tomato', linewidth=2,
                    label='Expected (CM null)', zorder=3)
            ax.fill_between(unique_degs, bin_exp - bin_std, bin_exp + bin_std,
                            color='tomato', alpha=0.2, label=std_label, zorder=2)
            ax.axhline(er_val, color='seagreen', linewidth=2, linestyle=':',
                       label=f'Erdős-Rényi baseline ({er_val:.2f})', zorder=3)

            ax.set_xlabel('Node Degree')
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.set_ylim(bottom=0, top=y_max)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend()

        plt.suptitle('Observed vs CM-Expected Similarity and Topology by Degree',
                     **PlotConfig.SUPTITLE_KW)
        _finalize_plot(fig, save_path, show)

