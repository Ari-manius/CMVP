"""
CMVP Core - Configuration Model Validated Projection

Main CMVP class supporting directed/undirected and weighted/unweighted networks.
Uses NEMtropy for null model fitting and NetworkX for similarity measures.
"""

import os
import numpy as np
import networkx as nx
from scipy import sparse
from typing import Tuple, Dict, Optional, Union, Callable
import sys
import io
import re
from tqdm import tqdm

# Numerical safety constants
MIN_PROB = 1e-300          # Minimum probability threshold (for log operations)
MIN_STRENGTH = 1e-15       # Minimum strength for geometric parameter estimation
PROB_UPPER_BOUND = 1.0     # Upper bound for probability parameters


def _get_directed_operation(mode: str) -> Callable:
    """Return the matrix operation function for a given directed mode.

    Parameters
    ----------
    mode : str
        Overlap mode: 'out-out', 'in-in', 'out-in', 'in-out'.

    Returns
    -------
    operation : callable
        Function that applies the directed mode operation to a matrix pair.
        Signature: operation(A, B) -> result_matrix
    """
    if mode == 'out-out':
        return lambda A, B: A @ B.T
    elif mode == 'in-in':
        return lambda A, B: A.T @ B
    elif mode == 'out-in':
        return lambda A, B: A @ B
    elif mode == 'in-out':
        return lambda A, B: (A @ B).T
    else:
        raise ValueError(f"Unknown directed mode: {mode}. Use 'out-out', 'in-in', 'out-in', or 'in-out'")


def _get_degree_vectors(A, mode: str, is_directed: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Get degree vectors for a given directed mode.

    Parameters
    ----------
    A : np.ndarray or sparse matrix
        Adjacency or probability matrix (n x n).
    mode : str
        Overlap mode: 'out-out', 'in-in', 'out-in', 'in-out'.
    is_directed : bool
        Whether the graph is directed.

    Returns
    -------
    deg_u : np.ndarray
        Degree vector for row nodes.
    deg_v : np.ndarray
        Degree vector for column nodes.
    """
    if not is_directed:
        deg = np.asarray(A.sum(axis=1)).flatten()
        return deg, deg

    out_deg = np.asarray(A.sum(axis=1)).flatten()
    in_deg = np.asarray(A.sum(axis=0)).flatten()

    # Mode 'u-v': row nodes use their u-neighborhood, column nodes their v-neighborhood
    mode_u, mode_v = mode.split('-')
    deg_u = out_deg if mode_u == 'out' else in_deg
    deg_v = out_deg if mode_v == 'out' else in_deg
    return deg_u, deg_v


def _compute_p_matrix_sparse(A_bin, parameters: dict) -> sparse.csr_matrix:
    """
    Compute p_matrix only for observed edges (sparse computation optimization).

    For large sparse graphs, this avoids computing full n×n matrices.
    Instead, only computes probabilities for edges that actually exist.

    Parameters
    ----------
    A_bin : sparse.csr_matrix
        Binary adjacency matrix (unweighted)
    parameters : dict
        Contains model-specific parameters (x, y, b_out, b_in, etc.)
        Expected keys depend on model type

    Returns
    -------
    p_sparse : sparse.csr_matrix
        Sparse p-matrix with entries only for observed edges
    """
    # Get edges from binary adjacency
    row_idx, col_idx = sparse.find(A_bin)[:2]

    if not len(row_idx):
        # No edges
        return sparse.csr_matrix(A_bin.shape, dtype=float)

    # Extract parameters (model-specific)
    x = parameters.get('x')
    y = parameters.get('y')
    b_out = parameters.get('b_out')
    b_in = parameters.get('b_in')

    # Compute p only for existing edges
    if b_out is not None and b_in is not None:
        # DECM model
        x_prod = x[row_idx] * x[col_idx]
        b_prod = b_out[row_idx] * b_in[col_idx]
        xy_prod = x_prod * b_prod
        p_vals = xy_prod / (1 - b_prod + xy_prod)
    elif y is not None:
        # ECM/CReMa model
        x_prod = x[row_idx] * x[col_idx]
        y_prod = y[row_idx] * y[col_idx]
        xy_prod = x_prod * y_prod
        p_vals = xy_prod / (1 - y_prod + xy_prod)
    else:
        # CM model (unweighted)
        x_prod = x[row_idx] * x[col_idx]
        p_vals = x_prod / (1 + x_prod)

    # Build sparse matrix
    p_sparse = sparse.csr_matrix(
        (p_vals, (row_idx, col_idx)), shape=A_bin.shape, dtype=np.float64
    )

    return p_sparse


def directed_matmul(A, mode, is_directed=True):
    """Compute overlap matrix and degree vectors for a given directed mode.

    Parameters
    ----------
    A : np.ndarray or sparse matrix
        Adjacency or probability matrix (n x n).
    mode : str
        Overlap mode: 'out-out', 'in-in', 'out-in', 'in-out'.
    is_directed : bool
        Whether the graph is directed. If False, mode is ignored.

    Returns
    -------
    CN : np.ndarray
        Overlap matrix (n x n). Diagonal is NOT zeroed.
    deg_u : np.ndarray
        1-D degree vector for row nodes.
    deg_v : np.ndarray
        1-D degree vector for column nodes.
    """
    if not is_directed:
        CN = A @ A
        deg = np.asarray(A.sum(axis=1)).flatten()
        return CN, deg, deg

    operation = _get_directed_operation(mode)
    CN = operation(A, A)
    deg_u, deg_v = _get_degree_vectors(A, mode, is_directed=True)
    return CN, deg_u, deg_v

SIMILARITY_MEASURES = ('common_neighbors', 'jaccard')  # Jaccard only for compute_similarity_matrix(), not validate_projection()

# ============================================================================
# CMVP CLASS
# ============================================================================

class CMVP:
    """
    Configuration-Model Validated Projection for network analysis.

    Supports:
    - Undirected/Directed graphs
    - Unweighted/Weighted networks
    - Multiple NEMtropy null models (UBCM, UECM, BiCM, DBCM, etc.)
    - Various similarity measures

    Key matrices:
    - p_matrix: Edge probabilities under null model
    - obs_sim_matrix: Observed similarity matrix
    - exp_sim_matrix: Expected similarity under null model
    - test_sim_matrix_zscore: (obs - exp) / std under null (two-sided normal statistic)
    - test_sim_matrix_effect: (obs - exp) / variance under null (tilted-log-odds effect
      size theta-hat; natural-parameter scale, opportunity/degree term divided out —
      distinct from significance, which conflates association strength with evidence)
    - test_sim_matrix_pvalue: P-values from significance testing
    - test_sim_matrix_sig: log(-log(p-value)) significance (signed for tail='both')
    """

    # Dense-array attributes saved/loaded as a single .npz (p_matrix/w_matrix included
    # here too; save() also mirrors them under their own keys for backward compatibility).
    _DENSE_ATTRS = ('p_matrix', 'w_matrix', 'x', 'x_out', 'x_in', 'y', 'y_out', 'y_in',
                    'obs_sim_matrix', 'exp_sim_matrix', 'exp_sim_matrix_std',
                    'test_sim_matrix_zscore', 'test_sim_matrix_effect',
                    'jaccard_matrix', 'lambda_params',
                    '_pairs', '_pvalues_array', '_pvalues_left')

    def __init__(self, graph: Union[nx.Graph, nx.DiGraph], seed: Optional[int] = None):
        """
        Initialize CMVP with a network.

        Parameters
        ----------
        graph : networkx.Graph or networkx.DiGraph
            The input network (can be directed or undirected, weighted or unweighted)
        seed : int, optional
            Random seed for reproducibility (default: None)

        Raises
        ------
        ValueError
            If the graph is invalid (empty, < 2 nodes, has self-loops)
        """
        # Input validation
        if graph is None:
            raise ValueError("Graph cannot be None")
        if len(graph.nodes()) == 0:
            raise ValueError("Graph is empty (no nodes)")
        if len(graph.nodes()) < 2:
            raise ValueError("Graph must have at least 2 nodes")
        if nx.number_of_selfloops(graph) > 0:
            raise ValueError("Graph contains self-loops. Remove them before using CMVP.")

        self.graph = graph
        self.nodes = list(graph.nodes())
        self.n = len(self.nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Detect graph properties
        self.is_directed = graph.is_directed()
        self.is_weighted = nx.is_weighted(graph)

        # Get adjacency matrix (weighted if graph is weighted)
        if self.is_weighted:
            self.A = nx.to_scipy_sparse_array(graph, nodelist=self.nodes, format='csr', weight='weight')
            # Also get binary version for some computations
            self.A_bin = (self.A > 0).astype(int)
        else:
            self.A = nx.to_scipy_sparse_array(graph, nodelist=self.nodes, format='csr')
            self.A_bin = self.A

        # Null model parameters (fitted lazily) — mirrors NEMtropy naming
        # Undirected: x (degree), y (strength)
        # Directed:   x_out, x_in (degree), y_out, y_in (strength)
        self.x = None
        self.x_out = None
        self.x_in = None
        self.y = None
        self.y_out = None
        self.y_in = None
        self.lambda_params = None
        self.p_matrix = None  # Edge probability matrix (null model, dense for backward compatibility)
        self.p_matrix_sparse = None  # Sparse version: only computed for observed edges (optimization)
        self.w_matrix = None  # Edge weight expectation matrix (for weighted models)
        self.convergence_history = None  # Convergence tracking
        self._nem_model = None  # Store fitted NEMtropy model for warm-starting

        # Configuration parameters (set during fit/validate)
        self._config = {
            'null_model': None,
            'null_model_method': None,
            'alpha': None,
            'similarity_measure': None,
            'correction_method': None,
            'transform': None,
        }

        # Result matrices (computed during validate_projection)
        self.obs_sim_matrix = None   # Observed similarity matrix
        self.exp_sim_matrix = None   # Expected similarity matrix under null
        self.exp_sim_matrix_std = None   # Std dev of expected similarity under null
        self.test_sim_matrix_zscore = None  # (obs - exp) / std under null
        self.test_sim_matrix_effect = None  # (obs - exp) / variance under null (theta-hat effect size)
        self.test_sim_matrix_pvalue = None  # P-value matrix
        self.test_sim_matrix_sig = None         # logneglog significance (signed for tail='both')
        self.jaccard_matrix = None            # Jaccard similarity matrix
        self.backbone = None                  # Current backbone (sparse)
        self.backbone_signed = None           # Signed backbone (+1/-1), only set for tail='both'
        self._pvalues_left = None             # Left-tail p-values, only set for tail='both'
        self._pairs = None                    # Candidate pairs array
        self._pvalues_array = None            # Raw p-values for pairs
        self._symmetric_directed = False      # True for out-out/in-in on directed graphs



    def fit_configuration_model(self, model: str = 'auto', method: str = 'fixed-point',
                               max_iter: int = 1000, verbose: bool = True, warm_start: bool = False,
                               initial_guess: Optional[str] = None):
        """
        Fit a configuration model using NEMtropy.

        Auto-selects appropriate model based on graph type if model='auto'.

        Parameters
        ----------
        model : str
            NEMtropy model type (default: 'auto' - auto-selects based on graph type)
            Options: 'auto', 'cm_exp', 'cm', 'ecm_exp', 'ecm', 'crema', 'dcm_exp', 'dcm', 'decm_exp', 'decm'
        method : str
            Solver method (default: 'fixed-point'). Options: 'fixed-point', 'newton', 'quasinewtonian'
        max_iter : int
            Maximum iterations (default: 1000)
        verbose : bool
            Print progress (default: True)
        warm_start : bool
            If True and a model is already fitted, use its parameters as initial guess (default: False).
            Dramatically speeds up refinement or model comparison on the same graph.
        initial_guess : str, optional
            Override the auto-selected NEMtropy initial guess ('degrees'/'strengths').
            E.g. 'random' to start the solver from a random point instead of the data-derived
            default — useful to test whether the solver converges to the same optimum
            regardless of starting point (solver/convergence-noise checks). Ignored if
            warm_start=True and a previous fit exists (that always uses 'from_previous').

        Returns
        -------
        self : CMVP
            Returns self for method chaining
        """
        # Auto-select model based on graph type
        if model == 'auto':
            if self.is_directed and self.is_weighted:
                model = 'decm_exp'
            elif self.is_directed:
                model = 'dcm_exp'
            elif self.is_weighted:
                model = 'ecm_exp'
            else:
                model = 'cm_exp'
            print(f"Auto-selected model: {model} (directed={self.is_directed}, weighted={self.is_weighted})")

        # Validate model-graph compatibility
        directed_models = {'dcm', 'dcm_exp', 'decm', 'decm_exp'}
        undirected_models = {'cm', 'cm_exp', 'ecm', 'ecm_exp', 'crema'}
        enhanced_models = {'ecm', 'ecm_exp', 'crema', 'decm', 'decm_exp'}

        if model in directed_models and not self.is_directed:
            raise ValueError(
                f"Model '{model}' requires a directed graph, but graph is undirected. "
                f"Use one of: {sorted(undirected_models)}"
            )
        if model in undirected_models and self.is_directed:
            raise ValueError(
                f"Model '{model}' requires an undirected graph, but graph is directed. "
                f"Use one of: {sorted(directed_models)}"
            )
        if model in enhanced_models and not self.is_weighted:
            raise ValueError(
                f"Model '{model}' requires a weighted graph, but graph is unweighted. "
                f"Use '{model.replace('ecm', 'cm').replace('decm', 'dcm')}' instead."
            )

        # Store configuration
        self._config['null_model'] = model
        self._config['null_model_method'] = method

        # Auto-select initial_guess based on model, unless explicitly overridden
        if initial_guess is None:
            initial_guess = 'strengths' if model in enhanced_models else 'degrees'

        # DECM fixed-point converges poorly; default to newton
        if 'decm' in model and method == 'fixed-point':
            method = 'newton'
            if verbose:
                print(f"Note: switched to method='newton' for {model} (fixed-point converges poorly)")

        # Check if warm-starting is viable
        if warm_start and self._nem_model is not None:
            if verbose:
                print(f"Fitting null model with NEMtropy (model={model}, method={method}, warm_start=True)...")
            initial_guess = 'from_previous'
        else:
            if verbose:
                print(f"Fitting null model with NEMtropy (model={model}, method={method})...")

        self._fit_nemtropy(model=model, method=method, max_iter=max_iter,
                          initial_guess=initial_guess, verbose=verbose)

        return self

    def _fit_nemtropy(self, model: str, method: str, max_iter: int,
                     initial_guess: str, verbose: bool):
        """
        Fit configuration model using the NEMtropy package.

        Handles both directed/undirected and weighted/unweighted graphs.
        Reference: https://github.com/nicoloval/NEMtropy
        """
        try:
            if self.is_directed:
                from NEMtropy import DirectedGraph as NemGraph
            else:
                from NEMtropy import UndirectedGraph as NemGraph
        except ImportError:
            raise ImportError(
                "NEMtropy package not found. Install with: pip install NEMtropy"
            )

        # Monkey-patch: fix NEMtropy's decm_exp Numba bug where degree
        # sequences are passed as object-dtype arrays instead of float64
        if model in ('decm', 'decm_exp') and not getattr(NemGraph, '_decm_patched', False):
            _original_set_args = NemGraph._set_args
            def _patched_set_args(self_nem, mdl):
                _original_set_args(self_nem, mdl)
                if mdl in ('decm', 'decm_exp'):
                    self_nem.args = tuple(np.asarray(a, dtype=np.float64) for a in self_nem.args)
            NemGraph._set_args = _patched_set_args
            NemGraph._decm_patched = True

        # Pass adjacency matrix directly to NEMtropy (faster than edge list)
        from scipy import sparse as sp
        A_dense = self.A.toarray() if sp.issparse(self.A) else np.array(self.A, dtype=float)
        nem_model = NemGraph(adjacency=A_dense)

        # Ensure strength sequence for weighted models
        if self.is_weighted:
            if not hasattr(nem_model, 'strength_sequence') or nem_model.strength_sequence is None:
                nem_model.strength_sequence = np.array(A_dense.sum(axis=1)).flatten()

        # Handle warm-starting: use parameters from previous fit
        if initial_guess == 'from_previous' and self._nem_model is not None:
            try:
                # Copy parameters from previously fitted model
                if hasattr(self._nem_model, 'x') and self._nem_model.x is not None:
                    nem_model.x = np.copy(self._nem_model.x)
                if hasattr(self._nem_model, 'y') and self._nem_model.y is not None:
                    nem_model.y = np.copy(self._nem_model.y)
                if hasattr(self._nem_model, 'b_out') and self._nem_model.b_out is not None:
                    nem_model.b_out = np.copy(self._nem_model.b_out)
                if hasattr(self._nem_model, 'b_in') and self._nem_model.b_in is not None:
                    nem_model.b_in = np.copy(self._nem_model.b_in)
                initial_guess = 'from_file'  # Tell NEMtropy to use loaded parameters
                print("  Using warm-start from previous fit")
            except Exception as e:
                print(f"  Warning: warm-start failed ({e}), falling back to default initial guess")
                initial_guess = 'strengths' if 'ecm' in model or 'decm' in model else 'degrees'

        # Always capture stdout: NEMtropy prints a trailing "solution error = ..." line
        # unconditionally (regardless of its own verbose flag), so this is the only way
        # to keep fit_configuration_model(verbose=False) actually silent. The captured
        # output is parsed into convergence_history either way; only our own derived
        # summary prints below are gated on `verbose`.
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            # Solve using NEMtropy
            nem_model.solve_tool(
                model=model,
                method=method,
                initial_guess=initial_guess,
                max_steps=max_iter,
                verbose=verbose
            )
        except Exception as e:
            # Restore stdout first
            sys.stdout = old_stdout

            # Provide helpful error messages for known issues
            if 'ecm' in model.lower() and 'TypingError' in str(type(e).__name__):
                raise ValueError(
                    f"NEMtropy's '{model}' model has a Numba compilation issue. "
                    f"This is a known bug in NEMtropy. Please use 'cm_exp' or 'cm' instead.\n"
                    f"Recommended: model='cm_exp' (canonical ensemble, soft constraints)"
                ) from e
            else:
                raise
        finally:
            # Restore stdout
            sys.stdout = old_stdout
            output = captured_output.getvalue()

        # Extract fitted parameters and compute p_matrix / w_matrix
        # NEMtropy parameter layout depends on model type:
        #   CM:   x (degree fitness, undirected)
        #   DCM:  x (out-degree fitness), y (in-degree fitness)
        #   ECM:  x (degree fitness), y (strength fitness, undirected)
        #   CReMa: x (degree fitness), strength parameters may use different attribute names
        #   DECM: x (out-degree), y (in-degree), b_out, b_in (strength)

        is_dcm = model in ('dcm', 'dcm_exp')
        is_decm = model in ('decm', 'decm_exp')
        is_ecm = model in ('ecm', 'ecm_exp')
        is_crema = model == 'crema'

        x = np.asarray(nem_model.x, dtype=float) if nem_model.x is not None else np.ones(self.n)

        if is_decm:
            # DECM: 4 parameter vectors (directed, weighted)
            x_out = np.asarray(nem_model.x, dtype=float)
            x_in = np.asarray(nem_model.y, dtype=float)
            b_out = np.asarray(nem_model.b_out, dtype=float)
            b_in = np.asarray(nem_model.b_in, dtype=float)

            ab = x_out[:, np.newaxis] * x_in[np.newaxis, :]
            cd = b_out[:, np.newaxis] * b_in[np.newaxis, :]
            cd = np.clip(cd, 0, 1 - MIN_STRENGTH)
            abcd = ab * cd

            self.p_matrix = abcd / (1 - cd + abcd)
            np.fill_diagonal(self.p_matrix, 0)

            self.w_matrix = self.p_matrix / (1 - cd)
            np.fill_diagonal(self.w_matrix, 0)

            self.x_out = x_out
            self.x_in = x_in
            self.y_out = b_out
            self.y_in = b_in

        elif is_dcm:
            # DCM: x = out-degree fitness, y = in-degree fitness (unweighted)
            # p_ij = x_i * y_j / (1 + x_i * y_j)
            x_out = np.asarray(nem_model.x, dtype=float)
            y_in = np.asarray(nem_model.y, dtype=float)
            xy = x_out[:, np.newaxis] * y_in[np.newaxis, :]
            xy = np.clip(xy, 0, 1e10)

            self.p_matrix = xy / (1.0 + xy)
            np.fill_diagonal(self.p_matrix, 0)

            self.w_matrix = None
            self.x_out = x_out
            self.x_in = y_in

        elif is_ecm:
            # ECM: x = degree fitness, y = strength fitness (undirected)
            # p_ij = (x_i*x_j*y_i*y_j) / (1 - y_i*y_j + x_i*x_j*y_i*y_j)
            y = np.asarray(nem_model.y, dtype=float)
            x_prod = x[:, np.newaxis] * x[np.newaxis, :]
            y_prod = y[:, np.newaxis] * y[np.newaxis, :]
            y_prod = np.clip(y_prod, 0, 1 - MIN_STRENGTH)
            xy_prod = x_prod * y_prod

            self.p_matrix = xy_prod / (1.0 - y_prod + xy_prod)
            np.fill_diagonal(self.p_matrix, 0)

            # E[W_ij] = p_ij / (1 - y_i*y_j)  (W: expected edge weight given
            # the edge exists — distinct from w_ij, the common-neighbor count)
            self.w_matrix = self.p_matrix / (1.0 - y_prod)
            np.fill_diagonal(self.w_matrix, 0)

            self.y = y

        elif is_crema:
            # CReMa: Separate constraints on binary structure and weights
            # CReMa stores strength parameters differently than ECM
            # Check for strength parameters (NEMtropy may store them as beta_strengths)

            # Try to get strength parameters from various possible attribute names
            y = None
            if hasattr(nem_model, 'y'):
                y = np.asarray(nem_model.y, dtype=float)
            elif hasattr(nem_model, 'beta_strengths'):
                y = np.asarray(nem_model.beta_strengths, dtype=float)
            elif hasattr(nem_model, 'beta'):
                y = np.asarray(nem_model.beta, dtype=float)
            else:
                # If no strength parameters found, CREMA might only provide binary probabilities
                # Fall back to using the UBCM (binary) model probabilities
                print("Warning: CReMa fitted but strength parameters not found. Using binary probabilities only.")
                self.p_matrix = x[:, np.newaxis] * x[np.newaxis, :] / (1.0 + x[:, np.newaxis] * x[np.newaxis, :])
                np.fill_diagonal(self.p_matrix, 0)
                self.w_matrix = None
                self.x = x
                return

            # Compute weighted probabilities using strength parameters
            x_prod = x[:, np.newaxis] * x[np.newaxis, :]
            y_prod = y[:, np.newaxis] * y[np.newaxis, :]
            y_prod = np.clip(y_prod, 0, 1 - MIN_STRENGTH)
            xy_prod = x_prod * y_prod

            self.p_matrix = xy_prod / (1.0 - y_prod + xy_prod)
            np.fill_diagonal(self.p_matrix, 0)

            # E[W_ij] = p_ij / (1 - y_i*y_j)  (W: expected edge weight given
            # the edge exists — distinct from w_ij, the common-neighbor count)
            self.w_matrix = self.p_matrix / (1.0 - y_prod)
            np.fill_diagonal(self.w_matrix, 0)

            self.y = y

        else:
            # CM (undirected, unweighted): p_ij = x_i*x_j / (1 + x_i*x_j)
            x_prod = x[:, np.newaxis] * x[np.newaxis, :]
            x_prod = np.clip(x_prod, 0, 1e10)

            self.p_matrix = x_prod / (1.0 + x_prod)
            np.fill_diagonal(self.p_matrix, 0)

            self.w_matrix = None

        # Store undirected degree parameter (CM/ECM/CReMa); directed models set x_out/x_in above
        if not (is_decm or is_dcm):
            self.x = x

        # Store fitted NEMtropy model for warm-starting on same graph
        self._nem_model = nem_model

        # Build convergence history from NEMtropy output
        # Parse the captured output to extract iteration history
        self.convergence_history = []

        lines = output.split('\n')

        # Parse line by line, collecting |f(x)| and diff for each iteration
        current_fx = None
        for line in lines:
            # Look for |f(x)| = value (constraint residual norm)
            fx_match = re.match(r'^\|f\(x\)\| = ([\d.e+-]+)', line)
            if fx_match:
                current_fx = float(fx_match.group(1))
                continue

            # Look for diff = value (solution change between iterations)
            diff_match = re.match(r'^diff = ([\d.e+-]+)', line)
            if diff_match and current_fx is not None:
                diff_val = float(diff_match.group(1))
                # Store both |f(x)| and diff
                self.convergence_history.append({
                    'iteration': len(self.convergence_history),
                    'max_error': current_fx,
                    'mean_error': current_fx,
                    'rmse': current_fx,
                    'diff': diff_val
                })
                current_fx = None  # Reset for next iteration

        # If we couldn't parse the history, fall back to final error only
        if not self.convergence_history:
            # Use binary adjacency for degree calculation (even for weighted graphs)
            degrees = np.array(self.A_bin.sum(axis=1)).flatten()
            expected_degrees = self.p_matrix.sum(axis=1)

            errors = degrees - expected_degrees
            max_error = np.max(np.abs(errors))
            mean_error = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))

            self.convergence_history.append({
                'iteration': 0,
                'max_error': max_error,
                'mean_error': mean_error,
                'rmse': rmse,
                'diff': 0.0  # No diff available in fallback case
            })
            if verbose:
                print(f"NEMtropy converged with max_error={max_error:.6e}, mean_error={mean_error:.6e}, rmse={rmse:.6e}")
        else:
            if verbose:
                print(f"Parsed {len(self.convergence_history)} iterations from NEMtropy output")
                print(f"Final |f(x)|: {self.convergence_history[-1]['max_error']:.6e}")
                print(f"Final diff: {self.convergence_history[-1]['diff']:.6e}")

        return nem_model

    def _compute_poisson_binomial_pvalues(self, P, pairs, observed_sim,
                                          mode, is_directed, tail, mid_p=False, verbose=True):
        """Exact Poisson-Binomial p-values via scipy.stats.poisson_binom.

        For each pair (u,v), CN(u,v) = Σ_k X_k where X_k ~ Bernoulli(p_k).
        P(X >= obs) = 1 - CDF(obs - 1)  [right tail]
        P(X <= obs) = CDF(obs)           [left tail]
        If mid_p, half the point mass at obs is subtracted to correct the
        conservative bias of the discrete p-value.
        """
        from scipy.stats import poisson_binom as pb_dist

        n_pairs = len(pairs)
        pvals = np.ones(n_pairs)

        for idx in tqdm(range(n_pairs), desc="Poisson-Binomial p-values", unit="pair", disable=not verbose):

            i, j = pairs[idx, 0], pairs[idx, 1]
            obs = int(observed_sim[i, j])

            # Extract per-intermediary Bernoulli probabilities π_k = p_ik * p_jk
            if not is_directed or mode == 'out-out':
                pk = P[i, :] * P[j, :]
            elif mode == 'in-in':
                pk = P[:, i] * P[:, j]
            elif mode == 'out-in':
                pk = P[i, :] * P[:, j]
            elif mode == 'in-out':
                pk = P[:, i] * P[j, :]
            else:
                pk = P[i, :] * P[j, :]

            pk = np.asarray(pk).flatten()
            # Drop degenerate Bernoulli terms; p=0 covers the zeroed diagonal
            # (self-pairs), p=1 terms are deterministic and not passed to poisson_binom
            pk = pk[(pk > 0) & (pk < 1)]

            if len(pk) == 0:
                pvals[idx] = 1.0
                continue

            if tail == 'right':
                # P(X >= obs) = 1 - P(X <= obs - 1) = 1 - CDF(obs - 1)
                pvals[idx] = 1 - pb_dist.cdf(obs - 1, pk) if obs > 0 else 1.0
                if mid_p:
                    pvals[idx] -= 0.5 * pb_dist.pmf(obs, pk)
            else:
                # P(X <= obs) = CDF(obs)
                pvals[idx] = pb_dist.cdf(obs, pk)
                if mid_p:
                    pvals[idx] -= 0.5 * pb_dist.pmf(obs, pk)

        return pvals

    # ------------------------------------------------------------------
    # validate_projection sub-stages
    # ------------------------------------------------------------------

    def _compute_expected_similarity(self, sim_name, pairs,
                                      directed_mode, weighted, test, tail, verbose=True):
        """Compute expected similarity and variance under the null model.

        Returns (expected_sim, null_std, use_poisson, pairs).
        For left-tail tests, pairs may be narrowed to those with expected > 0.
        """
        if verbose:
            print(f"Computing expected {sim_name} (analytical formula)...")
        P = self.p_matrix

        if weighted and sim_name == 'common_neighbors':
            # Weighted CN: analytical moments from ECM/DECM parameters
            # Under the null, W_ik = B_ik * G_ik where B ~ Bernoulli(p_ik)
            # and G ~ Geometric(q_ik) with q_ik derived from strength parameters.
            # E[W_ik] = w_matrix[i,k]
            # E[W_ik^2] = p_ik * (2 - q_ik) / q_ik^2
            # Var(W_ik) = E[W_ik^2] - E[W_ik]^2
            if verbose:
                print("  Computing analytical weighted CN moments...")

            W = self.w_matrix   # E[W_ik]
            P = self.p_matrix

            # Geometric parameter: q = 1 - (strength product)
            # Derived from NEMtropy's weighted models (W: expected edge weight,
            # self.w_matrix — distinct from w_ij, the common-neighbor count):
            # For ECM/CReMa: W_ij = p_ij / (1 - y_i*y_j), so q = (1 - y_i*y_j)
            # For DECM:      W_ij = p_ij / (1 - b_out_i*b_in_j), so q = (1 - b_out_i*b_in_j)

            if self.y_in is not None:
                # DECM: q_ij = 1 - y_out_i * y_in_j
                prod = self.y_out[:, np.newaxis] * self.y_in[np.newaxis, :]
                q_matrix = 1.0 - prod
                np.fill_diagonal(q_matrix, 1.0)
            elif self.y is not None:
                # ECM/CReMa: q_ij = 1 - y_i * y_j
                prod = self.y[:, np.newaxis] * self.y[np.newaxis, :]
                q_matrix = 1.0 - prod
                np.fill_diagonal(q_matrix, 1.0)
            else:
                # Fallback: compute numerically from p and w (less stable)
                q_matrix = np.divide(P, W, out=np.ones_like(P), where=W > 0)
                q_matrix = np.clip(q_matrix, MIN_STRENGTH, PROB_UPPER_BOUND)

            # Validate q values (should be in (0, 1] for valid geometric distribution)
            invalid_q = (q_matrix <= 0) | (q_matrix > 1.0)
            if np.any(invalid_q):
                n_invalid = np.sum(invalid_q)
                q_min, q_max = np.nanmin(q_matrix[q_matrix > 0]), np.nanmax(q_matrix)
                print(f"  ⚠️ Warning: {n_invalid} entries have invalid q (not in (0,1]). "
                      f"Range: [{q_min:.6f}, {q_max:.6f}]")

            # E[W^2] = p * (2 - q) / q^2
            E_W_sq = np.divide(P * (2.0 - q_matrix), q_matrix**2,
                               out=np.zeros_like(P), where=q_matrix > 0)
            np.fill_diagonal(E_W_sq, 0)

            # Var(W_ik) = E[W^2] - E[W]^2
            var_W = E_W_sq - W**2
            var_W = np.maximum(var_W, 0)

            # Expected weighted overlap: E[O_ij] = sum_k E[W_ik] * E[W_jk]
            expected_cn_w, _, _ = directed_matmul(W, directed_mode, self.is_directed)
            np.fill_diagonal(expected_cn_w, 0)
            expected_sim = expected_cn_w

            # Var(O_ij) = sum_k [ E[W_ik]^2 * Var(W_jk) + E[W_jk]^2 * Var(W_ik)
            #                     + Var(W_ik) * Var(W_jk) ]
            W_sq = W**2
            # Apply directed mode operation for variance computation
            op = _get_directed_operation(directed_mode) if self.is_directed else lambda A, B: A @ B
            t1 = op(W_sq, var_W)       # sum_k W_ik^2 * V_jk (or mode-dependent variant)
            t2 = op(var_W, W_sq)       # sum_k V_ik * W_jk^2 (or mode-dependent variant)
            t3 = op(var_W, var_W)      # sum_k V_ik * V_jk (or mode-dependent variant)

            variance = t1 + t2 + t3
            np.fill_diagonal(variance, 0)
            variance = np.maximum(variance, 0)
            null_std = np.sqrt(variance)
            use_poisson = False
        else:
            # Analytical path
            expected_cn, deg_u, deg_v = directed_matmul(P, directed_mode, self.is_directed)
            np.fill_diagonal(expected_cn, 0)

            # Common neighbors (only supported similarity for full validation)
            expected_sim = expected_cn
            use_poisson = (test == 'poisson')

            np.fill_diagonal(expected_sim, 0)

            # Compute variance analytically (also used for the z-score matrix)
            P_sq = P ** 2
            var_cn, _, _ = directed_matmul(P_sq, directed_mode, self.is_directed)
            variance = expected_cn - var_cn
            np.fill_diagonal(variance, 0)
            variance = np.maximum(variance, 0)
            null_std = np.sqrt(variance)

        # For left-tail only: narrow to pairs with expected > 0
        # (for 'both', all pairs with std > 0 are kept and filtered during z-score step)
        if tail == 'left':
            pairs = np.argwhere(expected_sim > 0)
            if not self.is_directed or self._symmetric_directed:
                pairs = pairs[pairs[:, 0] < pairs[:, 1]]
            if verbose:
                print(f"Found {len(pairs)} candidate pairs for dissimilarity test (expected > 0)")

        return expected_sim, null_std, use_poisson, pairs

    def _compute_pvalues(self, pairs, observed_sim, expected_sim, null_std,
                          use_poisson, test, tail, directed_mode, mid_p=False, verbose=True):
        """Compute p-values for common neighbors similarities."""
        tail_label = "right-tail (similarity)" if tail == 'right' else "left-tail (dissimilarity)"
        mid_p_label = ", mid-p" if mid_p and test in ('poisson', 'poisson-binomial') else ""
        if verbose:
            print(f"Computing p-values under null model ({tail_label}, test={test}{mid_p_label})...")
        from scipy.stats import poisson, norm

        pair_i, pair_j = pairs[:, 0], pairs[:, 1]
        obs_vals = observed_sim[pair_i, pair_j]
        exp_vals = expected_sim[pair_i, pair_j]
        pvalues_array = np.ones(len(pairs))

        if test == 'poisson-binomial':
            pvalues_array = self._compute_poisson_binomial_pvalues(
                self.p_matrix, pairs, observed_sim, directed_mode, self.is_directed, tail, mid_p, verbose=verbose)
        elif test == 'normal':
            std_vals = null_std[pair_i, pair_j]
            mask = (exp_vals > 0) & (std_vals > 0)
            # Continuity correction: CN is discrete (integer-valued), so shift the
            # boundary by 0.5 toward the mean before standardising.
            if tail == 'right':
                z = np.zeros(len(pairs))
                z[mask] = (obs_vals[mask] - 0.5 - exp_vals[mask]) / std_vals[mask]
                pvalues_array[mask] = 1 - norm.cdf(z[mask])
                no_var = ~mask
                pvalues_array[no_var] = np.where(obs_vals[no_var] <= exp_vals[no_var], 1.0, 0.0)
            else:
                z = np.zeros(len(pairs))
                z[mask] = (obs_vals[mask] + 0.5 - exp_vals[mask]) / std_vals[mask]
                pvalues_array[mask] = norm.cdf(z[mask])
                no_var = ~mask
                pvalues_array[no_var] = np.where(obs_vals[no_var] >= exp_vals[no_var], 1.0, 0.0)
        elif use_poisson:
            mask = exp_vals > 0
            large = mask & (exp_vals > 1000)
            small = mask & (exp_vals <= 1000)
            if tail == 'right':
                if small.any():
                    pvalues_array[small] = 1 - poisson.cdf(obs_vals[small] - 1, exp_vals[small])
                    if mid_p:
                        pvalues_array[small] -= 0.5 * poisson.pmf(obs_vals[small], exp_vals[small])
                if large.any():
                    z = (obs_vals[large] - exp_vals[large] - 0.5) / np.sqrt(exp_vals[large])
                    pvalues_array[large] = 1 - norm.cdf(z)
            else:
                if small.any():
                    pvalues_array[small] = poisson.cdf(obs_vals[small], exp_vals[small])
                    if mid_p:
                        pvalues_array[small] -= 0.5 * poisson.pmf(obs_vals[small], exp_vals[small])
                if large.any():
                    z = (obs_vals[large] - exp_vals[large] + 0.5) / np.sqrt(exp_vals[large])
                    pvalues_array[large] = norm.cdf(z)
        else:
            std_vals = null_std[pair_i, pair_j]
            mask = std_vals > 0
            z_scores = np.zeros(len(pairs))
            z_scores[mask] = (obs_vals[mask] - exp_vals[mask]) / std_vals[mask]
            if tail == 'right':
                pvalues_array[mask] = 1 - norm.cdf(z_scores[mask])
                no_std = ~mask
                pvalues_array[no_std] = np.where(obs_vals[no_std] <= exp_vals[no_std], 1.0, 0.0)
            else:
                pvalues_array[mask] = norm.cdf(z_scores[mask])
                no_std = ~mask
                pvalues_array[no_std] = np.where(obs_vals[no_std] >= exp_vals[no_std], 1.0, 0.0)

        return pvalues_array

    def _apply_correction(self, pvalues_array, alpha, correction, target_density):
        """Apply multiple testing correction or density-based thresholding.

        Returns boolean mask of significant pairs.
        """
        if target_density is not None:
            if not 0 < target_density <= 1:
                raise ValueError(f"target_density must be in (0, 1], got {target_density}")
            max_edges = len(pvalues_array)
            n_target_edges = min(int(np.ceil(target_density * max_edges)), max_edges)
            sorted_idx = np.argsort(pvalues_array)
            significant = np.zeros(len(pvalues_array), dtype=bool)
            significant[sorted_idx[:n_target_edges]] = True
            actual_density = np.sum(significant) / max_edges
            print(f"Density-based thresholding: target={target_density:.4f}, "
                  f"actual={actual_density:.4f} ({np.sum(significant)}/{max_edges} edges)")
        elif correction == 'fdr':
            print("Applying FDR (Benjamini-Hochberg) correction...")
            sorted_idx = np.argsort(pvalues_array)
            sorted_pvals = pvalues_array[sorted_idx]
            m = len(sorted_pvals)
            threshold_line = alpha * np.arange(1, m + 1) / m
            significant_mask = sorted_pvals <= threshold_line
            if np.any(significant_mask):
                max_k = np.where(significant_mask)[0][-1]
                threshold = sorted_pvals[max_k]
            else:
                threshold = 0  # no rejections: only exactly-zero p-values pass below
            significant = pvalues_array <= threshold
        elif correction == 'bonferroni':
            print("Applying Bonferroni correction...")
            significant = pvalues_array <= alpha / len(pvalues_array)
        elif correction == 'none':
            print("No multiple testing correction...")
            significant = pvalues_array <= alpha
        else:
            raise ValueError(f"Unknown correction method: {correction}. Use 'fdr', 'bonferroni', or 'none'.")

        return significant

    def _symmetrize(self, pair_i, pair_j, vals, symmetric):
        """Return (rows, cols, data), mirroring entries across the diagonal when symmetric."""
        if symmetric:
            return (np.concatenate([pair_i, pair_j]),
                    np.concatenate([pair_j, pair_i]),
                    np.concatenate([vals, vals]))
        return pair_i, pair_j, vals

    def _pairs_to_sparse(self, pair_i, pair_j, vals, symmetric):
        """Build a sparse n×n matrix from pair values, symmetrizing if needed."""
        rows, cols, data = self._symmetrize(pair_i, pair_j, vals, symmetric)
        return sparse.csr_matrix((data, (rows, cols)), shape=(self.n, self.n))

    def _transform_pvalues(self, pv, transform):
        """Convert raw p-values to the requested significance transform."""
        if transform == 'pvalue':
            return pv
        elif transform == 'neglog':
            return -np.log(np.clip(pv, MIN_PROB, PROB_UPPER_BOUND))
        elif transform == 'logneglog':
            return np.log1p(-np.log(np.clip(pv, MIN_PROB, PROB_UPPER_BOUND)))
        else:
            raise ValueError(f"Unknown transform: {transform}")

    def _apply_transform(self, pair_i, pair_j, pvalues_array, symmetric):
        """Build sparse test_sim_matrix_sig (logneglog of p-values) from tested pairs."""
        vals = self._transform_pvalues(pvalues_array, 'logneglog')
        self.test_sim_matrix_sig = self._pairs_to_sparse(pair_i, pair_j, vals, symmetric)

    # ------------------------------------------------------------------

    def validate_projection(self, similarity: str = 'common_neighbors',
                           directed_mode: str = 'out-out',
                           tail: str = 'right',
                           test: str = 'poisson',
                           mid_p: bool = False,
                           verbose: bool = True) -> 'CMVP':
        """
        Compute p-values for the projection. Call filter_backbone() afterwards to apply
        alpha / correction / target_density thresholds.

        Parameters
        ----------
        similarity : str
            Similarity measure (default: 'common_neighbors').
            Only 'common_neighbors' is supported for validation (analytical variance).
            Jaccard is available for comparison via compute_similarity_matrix().
        tail : str
            Type of test (default: 'right').
            'right': test for similarity (observed >> expected).
            'left': test for dissimilarity (observed << expected).
            'both': signed test — computes right- and left-tail p-values separately
            and keeps both (test_sim_matrix_pvalue / _pvalues_left), so the sign
            (which direction is significant) is preserved. Requires test='normal'
            (via the signed z-score) or test='poisson-binomial' (via separate exact
            one-sided p-values in each direction). 'poisson' is not supported for
            tail='both' — it has no natural signed/two-sided extension here.
        directed_mode : str
            For directed graphs (default: 'out-out'). Options: 'out-out', 'in-in', 'out-in', 'in-out'
        test : str
            Statistical test for p-value computation (default: 'poisson').
            'poisson': Poisson approximation (fast, default, good for sparse graphs).
            'normal': Normal approximation with continuity correction (better for dense graphs).
            'poisson-binomial': Exact Poisson-Binomial via DFT (most accurate, slower).
            Only applies when similarity is 'common_neighbors' (analytical path).
            For weighted null models (ecm/crema/decm), the weighted CN test is used
            and test is forced to 'normal'.
        mid_p : bool
            If True, use the mid-p variant (subtract half the point mass at the observed
            value) for 'poisson' and 'poisson-binomial' tests. CN is a discrete statistic,
            so the plain right-tail p-value P(X >= obs) is stochastically larger than
            Uniform[0,1] under the null (conservative). Mid-p corrects that bias and
            gives p-values closer to uniform, at the cost of no longer being exactly
            conservative (can slightly exceed the nominal alpha). No effect on 'normal'
            (which already applies its own continuity correction). Default: False.

        Notes
        -----
        test_sim_matrix_sig is always log(-log(p-value)) (logneglog, compresses
        the extreme range; higher = more significant). For tail='both', it is
        signed: each pair's transform is computed from whichever tail
        (right/similarity or left/dissimilarity) has the smaller p-value, with
        the result multiplied by +1 (right tail wins) or -1 (left tail wins) —
        analogous to the signed z-score, but usable with test='poisson-binomial'
        as well as test='normal'. Raw p-values remain available unchanged via
        test_sim_matrix_pvalue regardless of this.

        Returns
        -------
        self : CMVP
        """
        # Validate parameters
        if tail not in ('right', 'left', 'both'):
            raise ValueError(f"tail must be 'right', 'left', or 'both', got '{tail}'")
        if tail == 'both' and test not in ('normal', 'poisson-binomial'):
            raise ValueError("tail='both' requires test='normal' (z-score based) or "
                            "test='poisson-binomial' (exact, signed via left/right tail p-values)")
        if test not in ('poisson', 'normal', 'poisson-binomial'):
            raise ValueError(f"test must be 'poisson', 'normal', or 'poisson-binomial', got '{test}'")

        # Fit configuration model if not done yet
        if self.x is None and self.x_out is None:
            self.fit_configuration_model()

        # Weighted validation whenever the null model fitted strengths (ecm/crema/decm)
        weighted = self.w_matrix is not None
        if weighted and test != 'normal':
            if verbose:
                print(f"Note: weighted null model detected, switching test='{test}' to 'normal' (weighted CN test)")
            test = 'normal'

        # Validate similarity
        if similarity not in SIMILARITY_MEASURES:
            raise ValueError(f"Unknown similarity: {similarity}. Available: {list(SIMILARITY_MEASURES)}")
        if similarity == 'jaccard':
            raise ValueError("Jaccard is only available for comparison via compute_similarity_matrix(). "
                           "Use similarity='common_neighbors' for statistical validation.")

        # Store configuration
        self._config.update(directed_mode=directed_mode, tail=tail, test=test,
                            similarity_measure=similarity, transform='logneglog', mid_p=mid_p)

        # Compute observed similarities
        if verbose:
            print(f"Computing observed {similarity} similarities{' (weighted)' if weighted else ''}...")
        observed_sim = self.compute_similarity_matrix(measure=similarity, sparse_output=False, directed_mode=directed_mode)

        symmetric_directed = self.is_directed and directed_mode in ('out-out', 'in-in')
        self._symmetric_directed = symmetric_directed
        symmetric = not self.is_directed or symmetric_directed

        # Find candidate pairs for testing
        if tail == 'right':
            # Similarity: only test pairs with non-zero observed similarity
            pairs = np.argwhere(observed_sim > 0)
        else:
            # Dissimilarity or both: use all off-diagonal pairs initially
            pairs = np.argwhere(~np.eye(self.n, dtype=bool))

        if not self.is_directed or symmetric_directed:
            pairs = pairs[pairs[:, 0] < pairs[:, 1]]
        if verbose:
            print(f"Found {len(pairs)} candidate pairs for {'dissimilarity' if tail == 'left' else 'similarity' if tail == 'right' else 'signed'} test")

        # --- Stage 1: Expected similarity under null model ---
        expected_sim, null_std, use_poisson, pairs = self._compute_expected_similarity(
            similarity, pairs, directed_mode, weighted, test, tail, verbose=verbose)
        self.obs_sim_matrix = observed_sim
        self.exp_sim_matrix = expected_sim
        self.exp_sim_matrix_std = null_std

        # Z-score matrix: (obs - exp) / std for all pairs with variance under the null.
        # Equivalent to the tail='both' / test='normal' statistic.
        self.test_sim_matrix_zscore = np.divide(
            observed_sim - expected_sim, null_std,
            out=np.zeros_like(expected_sim), where=null_std > 0)

        # Effect size (theta-hat): (obs - exp) / variance, the Gaussian-approximation
        # natural-parameter estimate (one-step Newton update of the score equation
        # for an exponential-family tilt), i.e. z-score / std. Distinct from
        # significance: log p is roughly quadratic in the excess and carries an
        # extra factor of variance (theta-hat ~ z/std, -log p ~ z^2/2), so the two
        # decouple across pairs of different degree — same effect size, very
        # different p-values, or vice versa. Use this as the edge *weight*
        # (association strength/direction) and p-value/FDR as the edge *filter*
        # (which pairs survive); weighting by significance instead would re-inject
        # the degree/opportunity term UBCM validation was meant to strip out.
        null_var = null_std ** 2
        self.test_sim_matrix_effect = np.divide(
            observed_sim - expected_sim, null_var,
            out=np.zeros_like(expected_sim), where=null_var > 0)

        # Compute and store Jaccard similarity matrix
        self.jaccard_matrix = self.compute_similarity_matrix(
            'jaccard', sparse_output=False, directed_mode=directed_mode)

        if tail == 'both':
            pair_i, pair_j = pairs[:, 0], pairs[:, 1]

            if test == 'normal':
                from scipy.stats import norm as _norm
                obs_vals = observed_sim[pair_i, pair_j]
                exp_vals = expected_sim[pair_i, pair_j]
                std_vals = null_std[pair_i, pair_j]

                has_std = std_vals > 0
                z = np.zeros(len(pairs))
                z[has_std] = (obs_vals[has_std] - exp_vals[has_std]) / std_vals[has_std]

                p_right = np.ones(len(pairs))
                p_left  = np.ones(len(pairs))
                p_right[has_std] = 1 - _norm.cdf(z[has_std])
                p_left[has_std]  = _norm.cdf(z[has_std])
            else:  # poisson-binomial: exact one-sided p-values in each direction,
                   # kept separate (not merged into one two-sided p-value) so the
                   # sign — which of the two tails is significant — is preserved,
                   # same role the z-score's sign plays for test='normal'.
                p_right = self._compute_poisson_binomial_pvalues(
                    self.p_matrix, pairs, observed_sim, directed_mode, self.is_directed, 'right', mid_p, verbose=verbose)
                p_left = self._compute_poisson_binomial_pvalues(
                    self.p_matrix, pairs, observed_sim, directed_mode, self.is_directed, 'left', mid_p, verbose=verbose)

            self.test_sim_matrix_pvalue = self._pairs_to_sparse(pair_i, pair_j, p_right, symmetric)

            # Signed significance: whichever tail is more significant for each
            # pair, transformed and signed (+1 = right/similarity tail wins,
            # -1 = left/dissimilarity tail wins) — so test_sim_matrix_sig
            # reflects both directions instead of only the right tail.
            p_min = np.minimum(p_right, p_left)
            sign = np.where(p_right <= p_left, 1.0, -1.0)
            vals = sign * self._transform_pvalues(p_min, 'logneglog')
            self.test_sim_matrix_sig = self._pairs_to_sparse(pair_i, pair_j, vals, symmetric)

            self._pairs = pairs
            self._pvalues_array = p_right
            self._pvalues_left = p_left
            if verbose:
                print(f"Computed p-values for {len(pairs)} pairs (tail='both', test='{test}'). "
                      f"Call filter_backbone() to apply thresholds.")
            return self

        # --- Stage 2: P-values ---
        pvalues_array = self._compute_pvalues(
            pairs, observed_sim, expected_sim, null_std,
            use_poisson, test, tail, directed_mode, mid_p, verbose=verbose)

        pair_i, pair_j = pairs[:, 0], pairs[:, 1]
        if verbose:
            print(f"Computed p-values for {len(pvalues_array)} pairs. Call filter_backbone() to apply thresholds.")

        # --- Stage 3: P-value matrix and transform (sparse: only tested pairs) ---
        self.test_sim_matrix_pvalue = self._pairs_to_sparse(pair_i, pair_j, pvalues_array, symmetric)
        self._apply_transform(pair_i, pair_j, pvalues_array, symmetric)

        self._pairs = pairs
        self._pvalues_array = pvalues_array
        self._pvalues_left = None

        return self

    def filter_backbone(self, alpha: float = 0.01, correction: str = 'fdr',
                        target_density: float = None,
                        connected: bool = False) -> sparse.csr_matrix:
        """
        Re-filter the backbone from stored p-values without recomputing them.

        Apply alpha / correction / target_density thresholds to p-values computed by
        validate_projection(). Runs instantly since p-values are already stored.

        Parameters
        ----------
        alpha : float, optional
            Significance level. Required unless target_density or connected is set.
        correction : str
            Multiple testing correction: 'fdr', 'bonferroni', 'none' (default: 'fdr').
        target_density : float, optional
            Target backbone density in (0, 1]. Overrides alpha/correction.
        connected : bool
            If True, find minimum edges (by ascending p-value) for a single
            connected component. Overrides alpha and target_density. Also
            works after validate_projection(tail='both') — each pair is
            ranked/signed by whichever tail is more significant, producing
            a minimal connected *signed* backbone.

        Returns
        -------
        backbone : sparse.csr_matrix
            Re-filtered backbone adjacency matrix.
        """
        if self._pvalues_array is None:
            raise RuntimeError("Call validate_projection first to compute p-values.")

        # Signed re-filtering: rebuild backbone_signed from both p-value arrays
        if self._pvalues_left is not None:
            p_right = self._pvalues_array
            p_left  = self._pvalues_left
            pairs   = self._pairs
            pair_i, pair_j = pairs[:, 0], pairs[:, 1]
            symmetric = not self.is_directed or self._symmetric_directed

            if connected:
                return self._filter_connected_signed(pair_i, pair_j, p_right, p_left, symmetric)

            if target_density is not None:
                n_target = max(1, int(np.ceil(target_density * len(p_right))))
                # Rank each pair once by its *best* tail (p_right and p_left are
                # complementary / mirror-image rankings for test='normal', so
                # independently taking the top n_target of each tail makes the
                # two selections overlap once n_target exceeds half the pairs —
                # total collision, and an empty backbone, at target_density=1.0.
                # Ranking by min(p_right, p_left) once and signing by whichever
                # tail won makes that structurally impossible: each pair can
                # only ever be selected — and counted — once.
                p_min = np.minimum(p_right, p_left)
                sig = np.zeros(len(p_right), dtype=bool)
                sig[np.argsort(p_min)[:n_target]] = True
                sig_right = sig & (p_right <= p_left)
                sig_left  = sig & (p_right > p_left)
            elif alpha is not None:
                sig_right = self._apply_correction(p_right, alpha, correction, None)
                sig_left  = self._apply_correction(p_left,  alpha, correction, None)
            else:
                raise ValueError("Provide alpha, target_density, or connected=True.")

            conflict = sig_right & sig_left
            sig_right[conflict] = False
            sig_left[conflict]  = False

            def _signed_edges(mask, sign):
                si, sj = pair_i[mask], pair_j[mask]
                vals = np.full(len(si), sign, dtype=np.int8)
                return self._symmetrize(si, sj, vals, symmetric)

            r1, c1, d1 = _signed_edges(sig_right,  1)
            r2, c2, d2 = _signed_edges(sig_left,  -1)
            backbone_signed = sparse.csr_matrix(
                (np.concatenate([d1, d2]),
                 (np.concatenate([r1, r2]), np.concatenate([c1, c2]))),
                shape=(self.n, self.n))
            bb = backbone_signed.copy().astype(int)
            bb.data = np.abs(bb.data)
            self.backbone_signed = backbone_signed
            self.backbone = bb
            self._config['alpha'] = alpha
            self._config['correction_method'] = correction
            n_pos = int(sig_right.sum())
            n_neg = int(sig_left.sum())
            print(f"Signed backbone: {n_pos} positive, {n_neg} negative"
                  + (f", {int(conflict.sum())} conflicts dropped" if conflict.any() else ""))
            return backbone_signed

        pvalues_array = self._pvalues_array
        pairs = self._pairs
        pair_i, pair_j = pairs[:, 0], pairs[:, 1]

        if connected:
            backbone = self._filter_connected(pair_i, pair_j, pvalues_array)
        elif target_density is not None or alpha is not None:
            significant = self._apply_correction(pvalues_array, alpha, correction, target_density)
            if target_density is None:
                print(f"Found {np.sum(significant)} significant edges (alpha={alpha}, correction={correction})")
            backbone = self._build_backbone_from_mask(pair_i, pair_j, significant)
        else:
            raise ValueError("Provide alpha, target_density, or connected=True.")

        self.backbone = backbone
        if not connected:
            self._config['alpha'] = alpha
        self._config['correction_method'] = correction
        return backbone

    def _build_backbone_from_mask(self, pair_i, pair_j, mask):
        """Build sparse backbone matrix from a boolean mask over pairs."""
        sig_i, sig_j = pair_i[mask], pair_j[mask]
        symmetric = not self.is_directed or self._symmetric_directed
        return self._pairs_to_sparse(sig_i, sig_j, np.ones(len(sig_i), dtype=int), symmetric)

    def _min_connected_k(self, sorted_i, sorted_j):
        """Binary search for the minimum leading-k edges (by rank) forming a single
        connected component. Returns (k, total_pairs)."""
        max_k = len(sorted_i)

        def _is_connected(k):
            rows_k = sorted_i[:k]
            cols_k = sorted_j[:k]
            if not self.is_directed or self._symmetric_directed:
                rows_k = np.concatenate([rows_k, cols_k])
                cols_k = np.concatenate([sorted_j[:k], sorted_i[:k]])
            bb = sparse.csr_matrix(
                (np.ones(len(rows_k), dtype=int), (rows_k, cols_k)),
                shape=(self.n, self.n)
            )
            n_components = sparse.csgraph.connected_components(bb, directed=self.is_directed, return_labels=False)
            return n_components == 1

        if not _is_connected(max_k):
            raise RuntimeError(
                "Graph cannot be made connected with available edges. "
                "Consider using alpha-based thresholding instead."
            )

        lo, hi = 1, max_k
        while lo < hi:
            mid = (lo + hi) // 2
            if _is_connected(mid):
                hi = mid
            else:
                lo = mid + 1

        return lo, max_k

    def _filter_connected(self, pair_i, pair_j, pvalues_array):
        """Find minimum edges by ascending p-value for a connected backbone."""
        sorted_idx = np.argsort(pvalues_array)
        sorted_i, sorted_j = pair_i[sorted_idx], pair_j[sorted_idx]

        lo, max_k = self._min_connected_k(sorted_i, sorted_j)

        mask = np.zeros(len(pvalues_array), dtype=bool)
        mask[sorted_idx[:lo]] = True
        density = lo / max_k
        alpha_threshold = pvalues_array[sorted_idx[lo - 1]]
        self._config['alpha'] = alpha_threshold
        print(f"Min connected backbone: {lo}/{max_k} edges, density={density:.4f}, "
              f"alpha threshold={alpha_threshold:.6g}")
        return self._build_backbone_from_mask(pair_i, pair_j, mask)

    def _filter_connected_signed(self, pair_i, pair_j, p_right, p_left, symmetric):
        """Find minimum edges for a connected signed backbone.

        Each pair is ranked by whichever tail is more significant
        (p_min = min(p_right, p_left)), and takes that tail's sign — same
        one-p-value-per-pair logic as the non-connected signed path, just
        without an alpha/target_density cutoff. Connectivity only cares
        about which pairs are included, not their sign.
        """
        p_min = np.minimum(p_right, p_left)
        sign = np.where(p_right <= p_left, 1, -1).astype(np.int8)

        sorted_idx = np.argsort(p_min)
        sorted_i, sorted_j = pair_i[sorted_idx], pair_j[sorted_idx]

        lo, max_k = self._min_connected_k(sorted_i, sorted_j)

        keep = sorted_idx[:lo]
        si, sj, ssign = pair_i[keep], pair_j[keep], sign[keep]
        r, c, d = self._symmetrize(si, sj, ssign, symmetric)
        backbone_signed = sparse.csr_matrix((d, (r, c)), shape=(self.n, self.n))
        bb = backbone_signed.copy().astype(int)
        bb.data = np.abs(bb.data)
        self.backbone_signed = backbone_signed
        self.backbone = bb

        n_pos = int((ssign > 0).sum())
        n_neg = int((ssign < 0).sum())
        density = lo / max_k
        alpha_threshold = p_min[sorted_idx[lo - 1]]
        self._config['alpha'] = alpha_threshold
        print(f"Min connected signed backbone: {lo}/{max_k} edges "
              f"({n_pos} positive, {n_neg} negative), density={density:.4f}, "
              f"alpha threshold={alpha_threshold:.6g}")
        return backbone_signed

    @property
    def config(self) -> Dict:
        """
        Get current configuration.

        Returns
        -------
        config : dict
            Copy of the internal config: null_model, null_model_method, alpha,
            similarity_measure, correction_method, transform, and (once
            validate_projection has run) directed_mode, tail, test, mid_p.
        """
        return self._config.copy()

    @property
    def results(self) -> Dict:
        """
        Get computed results matrices.

        Returns
        -------
        results : dict
            Matrices that have been computed, keyed by attribute name
            (p_matrix, w_matrix, obs/exp/test_sim_matrix_*, jaccard_matrix).
        """
        results = {}
        if self.p_matrix is not None:
            results['p_matrix'] = self.p_matrix
        if self.w_matrix is not None:
            results['w_matrix'] = self.w_matrix

        if self.obs_sim_matrix is not None:
            results['obs_sim_matrix'] = self.obs_sim_matrix

        if self.exp_sim_matrix is not None:
            results['exp_sim_matrix'] = self.exp_sim_matrix
        if self.exp_sim_matrix_std is not None:
            results['exp_sim_matrix_std'] = self.exp_sim_matrix_std
        if self.test_sim_matrix_zscore is not None:
            results['test_sim_matrix_zscore'] = self.test_sim_matrix_zscore
        if self.test_sim_matrix_effect is not None:
            results['test_sim_matrix_effect'] = self.test_sim_matrix_effect

        if self.test_sim_matrix_pvalue is not None:
            results['test_sim_matrix_pvalue'] = self.test_sim_matrix_pvalue
        if self.test_sim_matrix_sig is not None:
            results['test_sim_matrix_sig'] = self.test_sim_matrix_sig

        if self.jaccard_matrix is not None:
            results['jaccard_matrix'] = self.jaccard_matrix
        return results

    def compute_p_matrix_sparse(self) -> sparse.csr_matrix:
        """
        Compute sparse p_matrix (only for observed edges).

        This is an optimization for large sparse graphs where the full n×n dense
        p_matrix would be wasteful. Only computes probabilities for edges that
        actually exist in the observed graph.

        Returns
        -------
        p_sparse : sparse.csr_matrix
            Sparse edge probability matrix, only with entries for observed edges

        Raises
        ------
        ValueError
            If model is not yet fitted
        """
        if self.p_matrix is None:
            raise ValueError("Must fit_configuration_model() first")

        # Collect parameters from fitted model
        if self.x_out is not None:
            params = {'x': self.x_out}
            if self.x_in is not None:
                params['y'] = self.x_in
            if self.y_out is not None:
                params['b_out'] = self.y_out
            if self.y_in is not None:
                params['b_in'] = self.y_in
        else:
            params = {'x': self.x}
            if self.y is not None:
                params['y'] = self.y

        # Compute sparse version
        p_sparse = _compute_p_matrix_sparse(self.A_bin, params)
        self.p_matrix_sparse = p_sparse

        # Report memory savings
        n = self.n
        nnz = p_sparse.nnz
        dense_size_mb = n * n * 8 / 1e6
        sparse_size_mb = nnz * 16 / 1e6  # Both value and index
        savings_pct = 100 * (1 - sparse_size_mb / dense_size_mb)

        print(f"Sparse p_matrix: {nnz}/{n*n} entries ({100*nnz/(n*n):.2f}%)")
        print(f"Memory: {sparse_size_mb:.1f} MB sparse vs {dense_size_mb:.1f} MB dense "
              f"(saves {savings_pct:.1f}%)")

        return p_sparse

    def compute_similarity_matrix(self, measure: str = 'common_neighbors',
                                  sparse_output: bool = True,
                                  directed_mode: str = 'out-out') -> Union[np.ndarray, sparse.csr_matrix]:
        """
        Compute similarity matrix using matrix multiplication.

        Parameters
        ----------
        measure : str
            Similarity measure. Options: 'common_neighbors' (for validation), 'jaccard' (comparison/visualization only).
        sparse_output : bool
            If True, return sparse matrix (default). Otherwise return dense numpy array.
        directed_mode : str
            For directed graphs: 'out-out', 'in-in', 'out-in', or 'in-out'.

        Notes
        -----
        Common neighbors uses the weighted adjacency when a weighted null model
        (ecm/crema/decm) has been fitted, binary otherwise. Jaccard is always binary.

        Returns
        -------
        similarity_matrix : np.ndarray or sparse.csr_matrix
        """
        if measure not in SIMILARITY_MEASURES:
            raise ValueError(f"Unknown measure: {measure}. Available: {list(SIMILARITY_MEASURES)}")

        if measure == 'common_neighbors':
            A = self.A if self.w_matrix is not None else self.A_bin
            W, _, _ = directed_matmul(A, directed_mode, self.is_directed)
            if sparse.issparse(W):
                W = W.tocsr()
                W.setdiag(0)
                W.eliminate_zeros()
                W = W.astype(np.float64)
                if not sparse_output:
                    W = W.toarray()
            else:
                np.fill_diagonal(W, 0)
                W = W.astype(np.float64)
                if sparse_output:
                    W = sparse.csr_matrix(W)
            return W

        else:  # jaccard
            A = self.A_bin
            CN, deg_u, deg_v = directed_matmul(A, directed_mode, self.is_directed)
            if sparse.issparse(CN):
                CN = CN.toarray()
            np.fill_diagonal(CN, 0)
            union = deg_u[:, None] + deg_v[None, :] - CN
            jaccard = np.divide(CN, union, out=np.zeros_like(CN, dtype=float), where=union > 0)
            np.fill_diagonal(jaccard, 0)
            if sparse_output:
                return sparse.csr_matrix(jaccard)
            return jaccard

    def to_networkx(self, backbone: sparse.csr_matrix = None,
                    weight: str = 'sig') -> Union[nx.Graph, nx.DiGraph]:
        """
        Convert backbone to NetworkX graph with node attributes copied from original.

        Parameters
        ----------
        backbone : sparse.csr_matrix, optional
            The backbone adjacency matrix. If None, uses stored self.backbone.
        weight : str
            Which matrix becomes the edge 'weight' attribute (default: 'sig'):
            'sig': test_sim_matrix_sig (logneglog-transformed significance).
            'effect': test_sim_matrix_effect (theta-hat effect size — association
            strength/direction with the degree/opportunity term divided out).
            Both 'sig' and 'effect' are always attached as their own named edge
            attributes too (when computed), regardless of this choice — this only
            controls which one is duplicated onto 'weight' for tools (e.g. Louvain)
            that read a graph's generic 'weight' attribute by default. Use 'effect'
            to avoid re-weighting community detection by significance, which would
            re-inject the degree term UBCM validation already divided out.

        Returns
        -------
        G_backbone : networkx.Graph or networkx.DiGraph
            The backbone network with node attributes and edge weights.
        """
        if backbone is None:
            backbone = self.backbone
        if backbone is None:
            raise ValueError("No backbone available. Call validate_projection() first or pass backbone explicitly.")

        return self._backbone_to_nx(backbone, weight=weight)

    def to_networkx_signed(self, weight: str = 'sig') -> Union[nx.Graph, nx.DiGraph]:
        """
        Convert signed backbone to NetworkX graph with 'sign' (+1/-1) and 'weight' edge attributes.
        Only available after validate_projection(tail='both') or filter_backbone() on a 'both' result.

        Parameters
        ----------
        weight : str
            Which matrix becomes the edge 'weight' attribute — see to_networkx().
        """
        if self.backbone_signed is None:
            raise ValueError("No signed backbone. Call validate_projection(tail='both') first.")

        return self._backbone_to_nx(self.backbone_signed, signed=True, weight=weight)

    def _backbone_to_nx(self, matrix: sparse.csr_matrix, signed: bool = False,
                        weight: str = 'sig') -> Union[nx.Graph, nx.DiGraph]:
        """Build a NetworkX graph from a sparse backbone matrix.

        Node attributes are copied from the original graph. Both test_sim_matrix_sig
        (significance) and test_sim_matrix_effect (theta-hat effect size) are
        attached as their own 'sig'/'effect' edge attributes when available, and
        `weight` selects which of the two is duplicated onto the generic 'weight'
        attribute — significance decides which edges survive (already applied via
        the backbone filter), theta-hat is the association strength/direction with
        the degree term divided out, meant to be usable as the actual edge weight
        for downstream community detection etc. instead of re-weighting by
        significance. If signed, matrix values (+1/-1) are stored as a 'sign' edge
        attribute.
        """
        if weight not in ('sig', 'effect'):
            raise ValueError(f"weight must be 'sig' or 'effect', got '{weight}'")

        sig_vals = self.test_sim_matrix_sig
        effect_vals = self.test_sim_matrix_effect
        weight_source = sig_vals if weight == 'sig' else effect_vals

        # Undirected output for undirected input or symmetric directed modes
        symmetric = not self.is_directed or self._symmetric_directed
        G = nx.Graph() if symmetric else nx.DiGraph()

        for node in self.nodes:
            G.add_node(node, **dict(self.graph.nodes[node]))

        rows, cols = matrix.nonzero()
        for i, j in zip(rows, cols):
            if i == j:
                continue
            if symmetric and i > j:  # add each undirected edge once
                continue
            attrs = {'weight': float(weight_source[i, j]) if weight_source is not None else 1.0}
            if sig_vals is not None:
                attrs['sig'] = float(sig_vals[i, j])
            if effect_vals is not None:
                attrs['effect'] = float(effect_vals[i, j])
            if signed:
                attrs['sign'] = int(matrix[i, j])
            G.add_edge(self.nodes[i], self.nodes[j], **attrs)

        return G

    def save(self, path: str):
        """
        Save fitted CMVP object to disk (numpy .npz + metadata).

        Saves all computed matrices and configuration so that
        validate_projection does not need to be re-run.

        Parameters
        ----------
        path : str
            File path (without extension). Creates <path>.npz and <path>_graph.graphml.
        """
        import json

        # Collect arrays
        arrays = {attr: getattr(self, attr) for attr in self._DENSE_ATTRS
                  if getattr(self, attr) is not None}
        if self.test_sim_matrix_pvalue is not None:
            sparse.save_npz(f"{path}_pvalue.npz", self.test_sim_matrix_pvalue)
        if self.test_sim_matrix_sig is not None:
            sparse.save_npz(f"{path}_sig.npz", self.test_sim_matrix_sig)

        # Save arrays
        np.savez_compressed(f"{path}.npz", **arrays)

        # Save graph
        graph_path = f"{path}_graph.graphml"
        nx.write_graphml(self.graph, graph_path)

        # Save metadata
        meta = {
            'nodes': self.nodes,
            'seed': self.seed,
            'is_directed': self.is_directed,
            'is_weighted': self.is_weighted,
            'config': self._config,
            'symmetric_directed': self._symmetric_directed,
        }
        with open(f"{path}_meta.json", 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        print(f"Saved CMVP to {path} (.npz, _graph.graphml, _meta.json)")

    @classmethod
    def load(cls, path: str) -> 'CMVP':
        """
        Load a saved CMVP object from disk.

        Parameters
        ----------
        path : str
            File path (without extension), same as used in save().

        Returns
        -------
        CMVP
            Restored CMVP object with all matrices.
        """
        import json

        # Load metadata
        with open(f"{path}_meta.json", 'r') as f:
            meta = json.load(f)

        # Load graph
        graph_path = f"{path}_graph.graphml"
        if meta['is_directed']:
            G = nx.read_graphml(graph_path, node_type=str)
            G = nx.DiGraph(G)
        else:
            G = nx.read_graphml(graph_path, node_type=str)
            G = nx.Graph(G)

        # Create instance
        obj = cls(G, seed=meta.get('seed'))
        obj._config = meta.get('config', {})
        obj._symmetric_directed = meta.get('symmetric_directed', False)

        # Load arrays
        data = np.load(f"{path}.npz", allow_pickle=False)

        for attr in cls._DENSE_ATTRS:
            if attr in data:
                setattr(obj, attr, data[attr])

        pvalue_path = f"{path}_pvalue.npz"
        sig_path = f"{path}_sig.npz"
        if os.path.exists(pvalue_path):
            obj.test_sim_matrix_pvalue = sparse.load_npz(pvalue_path)
        if os.path.exists(sig_path):
            obj.test_sim_matrix_sig = sparse.load_npz(sig_path)

        # Rebuild backbone from stored pairs + pvalues if available
        if obj._pvalues_array is not None and obj._pairs is not None:
            config = obj._config
            alpha = config.get('alpha')
            correction = config.get('correction_method', 'fdr')
            if alpha is not None:
                obj.filter_backbone(alpha=alpha, correction=correction)

        print(f"Loaded CMVP from {path} (n={obj.n}, fitted={'yes' if obj.p_matrix is not None else 'no'}, "
              f"validated={'yes' if obj.test_sim_matrix_pvalue is not None else 'no'})")
        return obj


def cmvp_backbone(G: Union[nx.Graph, nx.DiGraph], alpha: float = 0.01, similarity: str = 'common_neighbors',
                  correction: str = 'fdr', model: str = 'auto', method: str = 'fixed-point',
                  max_iter: int = 1000, verbose: bool = False,
                  directed_mode: str = 'out-out', seed: Optional[int] = None,
                  test: str = 'poisson') -> 'CMVP':
    """
    One-shot function to extract CMVP backbone from a network.

    Returns the full CMVP object with all computed matrices accessible
    (backbone, test_sim_matrix_pvalue, obs_sim_matrix, exp_sim_matrix, etc.).

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input network (directed or undirected, weighted or unweighted)
    alpha : float
        Significance level (default: 0.01)
    similarity : str
        Similarity measure (default: 'common_neighbors', the only one supported for validation).
    correction : str
        Multiple testing correction (default: 'fdr'). Options: 'fdr', 'bonferroni', 'none'
    model : str
        NEMtropy model (default: 'auto'). Auto-selects based on graph type.
    method : str
        Solver method (default: 'fixed-point'). Options: 'fixed-point', 'newton', 'quasinewtonian'
    max_iter : int
        Maximum iterations (default: 1000)
    verbose : bool
        Print progress (default: False)
    directed_mode : str
        For directed graphs (default: 'out-out'). Options: 'out-out', 'in-in', 'out-in', 'in-out'
    seed : int, optional
        Random seed for reproducibility (default: None)
    test : str
        Statistical test (default: 'poisson'). Options: 'poisson', 'normal', 'poisson-binomial'

    Returns
    -------
    cmvp : CMVP
        The fitted and validated CMVP object.
    """
    cmvp = CMVP(G, seed=seed)
    cmvp.fit_configuration_model(model=model, method=method, max_iter=max_iter, verbose=verbose)
    cmvp.validate_projection(similarity=similarity, directed_mode=directed_mode,
                             test=test)
    cmvp.filter_backbone(alpha=alpha, correction=correction)

    return cmvp
