"""
Regression harness for cmvp.core.CMVP: exercises the full parameter
matrix (model auto-selection x directed/weighted x test x tail x mid_p x
correction x filter mode) so future refactors of core.py / utils / addons
have something concrete to check outputs against.
"""

import numpy as np
import networkx as nx
import pytest

from cmvp import CMVP
from conftest import GRAPH_BUILDERS, EXPECTED_AUTO_MODEL


# ---------------------------------------------------------------------------
# Model auto-selection + basic fit/validate/filter/export, for every graph type
# ---------------------------------------------------------------------------

def test_auto_model_selection(small_graph, graph_kind):
    cmvp = CMVP(small_graph, seed=1)
    cmvp.fit_configuration_model(model='auto')
    assert cmvp._config['null_model'] == EXPECTED_AUTO_MODEL[graph_kind]
    assert cmvp.p_matrix is not None
    assert cmvp.p_matrix.shape == (cmvp.n, cmvp.n)
    assert np.all((cmvp.p_matrix >= 0) & (cmvp.p_matrix <= 1))


def test_full_pipeline_default_params(small_graph):
    cmvp = CMVP(small_graph, seed=1)
    cmvp.fit_configuration_model()
    cmvp.validate_projection()
    cmvp.filter_backbone(alpha=0.5, correction='none')  # generous alpha: exercise a non-empty path
    G_bb = cmvp.to_networkx()
    assert set(G_bb.nodes()) == set(small_graph.nodes())
    assert nx.density(G_bb) <= nx.density(small_graph) + 1e-9


# ---------------------------------------------------------------------------
# test x tail combinations
# ---------------------------------------------------------------------------

VALID_TEST_TAIL = [
    ('poisson', 'right', False),
    ('poisson', 'right', True),   # mid_p
    ('poisson', 'left', False),
    ('normal', 'right', False),
    ('normal', 'left', False),
    ('normal', 'both', False),
    ('poisson-binomial', 'right', False),
    ('poisson-binomial', 'left', True),
    ('poisson-binomial', 'both', False),
]


@pytest.mark.parametrize('test,tail,mid_p', VALID_TEST_TAIL)
def test_test_tail_mid_p_combinations(uu_graph, test, tail, mid_p):
    cmvp = CMVP(uu_graph, seed=1)
    cmvp.fit_configuration_model()
    cmvp.validate_projection(test=test, tail=tail, mid_p=mid_p)

    pvals = cmvp._pvalues_array
    assert pvals is not None
    assert np.all((pvals >= 0) & (pvals <= 1))
    if tail == 'both':
        assert cmvp._pvalues_left is not None
        assert np.all((cmvp._pvalues_left >= 0) & (cmvp._pvalues_left <= 1))


def test_weighted_null_model_forces_normal_test(uw_graph):
    """Weighted null models (ecm/decm) only support the weighted CN test ('normal')."""
    cmvp = CMVP(uw_graph, seed=1)
    cmvp.fit_configuration_model()  # auto -> ecm_exp
    cmvp.validate_projection(test='poisson')  # should silently switch to 'normal'
    assert np.all((cmvp._pvalues_array >= 0) & (cmvp._pvalues_array <= 1))


def test_tail_both_requires_normal_or_poisson_binomial(uu_graph):
    cmvp = CMVP(uu_graph, seed=1)
    cmvp.fit_configuration_model()
    with pytest.raises(ValueError):
        cmvp.validate_projection(test='poisson', tail='both')


def test_invalid_tail_raises(uu_graph):
    cmvp = CMVP(uu_graph, seed=1)
    cmvp.fit_configuration_model()
    with pytest.raises(ValueError):
        cmvp.validate_projection(tail='sideways')


def test_invalid_test_raises(uu_graph):
    cmvp = CMVP(uu_graph, seed=1)
    cmvp.fit_configuration_model()
    with pytest.raises(ValueError):
        cmvp.validate_projection(test='bogus')


def test_mid_p_pvalues_leq_plain(uu_graph):
    """Mid-p corrects conservative bias, so mid-p p-values should be <= plain ones."""
    cmvp_plain = CMVP(uu_graph, seed=1)
    cmvp_plain.fit_configuration_model()
    cmvp_plain.validate_projection(test='poisson', mid_p=False)

    cmvp_midp = CMVP(uu_graph, seed=1)
    cmvp_midp.fit_configuration_model()
    cmvp_midp.validate_projection(test='poisson', mid_p=True)

    assert np.all(cmvp_midp._pvalues_array <= cmvp_plain._pvalues_array + 1e-12)


# ---------------------------------------------------------------------------
# Directed modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('directed_mode', ['out-out', 'in-in', 'out-in', 'in-out'])
def test_directed_modes(du_graph, directed_mode):
    cmvp = CMVP(du_graph, seed=1)
    cmvp.fit_configuration_model()
    cmvp.validate_projection(directed_mode=directed_mode)
    assert cmvp._pvalues_array is not None
    assert np.all((cmvp._pvalues_array >= 0) & (cmvp._pvalues_array <= 1))


# ---------------------------------------------------------------------------
# filter_backbone: correction methods, target_density, connected, re-filtering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('correction', ['fdr', 'bonferroni', 'none'])
def test_correction_methods(uu_graph, correction):
    cmvp = CMVP(uu_graph, seed=1)
    cmvp.fit_configuration_model()
    cmvp.validate_projection()
    backbone = cmvp.filter_backbone(alpha=0.5, correction=correction)
    assert backbone.shape == (cmvp.n, cmvp.n)


def test_bonferroni_stricter_than_none(uu_graph):
    cmvp = CMVP(uu_graph, seed=1)
    cmvp.fit_configuration_model()
    cmvp.validate_projection()
    bb_none = cmvp.filter_backbone(alpha=0.5, correction='none')
    bb_bonf = cmvp.filter_backbone(alpha=0.5, correction='bonferroni')
    assert bb_bonf.nnz <= bb_none.nnz


def test_target_density_filter(uu_graph):
    cmvp = CMVP(uu_graph, seed=1)
    cmvp.fit_configuration_model()
    cmvp.validate_projection()
    n_pairs = len(cmvp._pvalues_array)
    target = 0.3
    cmvp.filter_backbone(target_density=target)
    n_sig = cmvp.backbone.nnz // 2  # symmetric, undirected
    assert abs(n_sig - round(target * n_pairs)) <= 1


def test_connected_filter_undirected(uu_graph):
    cmvp = CMVP(uu_graph, seed=1)
    cmvp.fit_configuration_model()
    cmvp.validate_projection()
    cmvp.filter_backbone(connected=True)
    G_bb = cmvp.to_networkx()
    assert nx.is_connected(G_bb)


def test_connected_filter_directed(du_graph):
    cmvp = CMVP(du_graph, seed=1)
    cmvp.fit_configuration_model()
    # out-out/in-in directed modes are symmetric -> to_networkx() yields an
    # undirected nx.Graph, so use out-in (asymmetric) to get a real DiGraph.
    cmvp.validate_projection(directed_mode='out-in')
    cmvp.filter_backbone(connected=True)
    G_bb = cmvp.to_networkx()
    assert isinstance(G_bb, nx.DiGraph)
    assert nx.is_weakly_connected(G_bb)


def test_refilter_without_refit_is_cheap_and_consistent(uu_graph):
    """filter_backbone() re-thresholds stored p-values; re-running validate_projection
    should not be required, and the stored p-values must not change between calls."""
    cmvp = CMVP(uu_graph, seed=1)
    cmvp.fit_configuration_model()
    cmvp.validate_projection()
    pvals_before = cmvp._pvalues_array.copy()

    cmvp.filter_backbone(alpha=0.9, correction='none')
    n_loose = cmvp.backbone.nnz
    cmvp.filter_backbone(alpha=0.01, correction='none')
    n_strict = cmvp.backbone.nnz

    assert n_strict <= n_loose
    np.testing.assert_array_equal(pvals_before, cmvp._pvalues_array)


def test_signed_backbone_tail_both(uu_graph):
    cmvp = CMVP(uu_graph, seed=1)
    cmvp.fit_configuration_model()
    cmvp.validate_projection(test='normal', tail='both')
    cmvp.filter_backbone(alpha=0.5, correction='none')
    assert cmvp.backbone_signed is not None
    assert set(np.unique(cmvp.backbone_signed.data)) <= {-1, 1}

    # connected=True must also work in signed mode
    cmvp.filter_backbone(connected=True)
    assert cmvp.backbone_signed is not None


def test_filter_backbone_before_validate_raises(uu_graph):
    cmvp = CMVP(uu_graph, seed=1)
    cmvp.fit_configuration_model()
    with pytest.raises(RuntimeError):
        cmvp.filter_backbone(alpha=0.05)


def test_filter_backbone_requires_a_criterion(uu_graph):
    cmvp = CMVP(uu_graph, seed=1)
    cmvp.fit_configuration_model()
    cmvp.validate_projection()
    with pytest.raises(ValueError):
        cmvp.filter_backbone(alpha=None)


# ---------------------------------------------------------------------------
# validate_projection auto-fits if fit_configuration_model was never called
# ---------------------------------------------------------------------------

def test_validate_projection_auto_fits(uu_graph):
    cmvp = CMVP(uu_graph, seed=1)
    assert cmvp.x is None
    cmvp.validate_projection()
    assert cmvp.x is not None


# ---------------------------------------------------------------------------
# Export: to_networkx / to_networkx_signed / compute_similarity_matrix
# ---------------------------------------------------------------------------

def test_to_networkx_signed(uu_graph):
    cmvp = CMVP(uu_graph, seed=1)
    cmvp.fit_configuration_model()
    cmvp.validate_projection(test='normal', tail='both')
    cmvp.filter_backbone(alpha=0.5, correction='none')
    G_signed = cmvp.to_networkx_signed()
    weights = [d['sig'] for _, _, d in G_signed.edges(data=True)]
    assert set(np.sign(weights)) <= {-1.0, 0.0, 1.0}


def test_compute_similarity_matrix_jaccard(uu_graph):
    cmvp = CMVP(uu_graph, seed=1)
    jac = cmvp.compute_similarity_matrix('jaccard')
    assert jac.shape == (cmvp.n, cmvp.n)
    jac_dense = jac.toarray() if hasattr(jac, 'toarray') else jac
    assert np.all((jac_dense >= 0) & (jac_dense <= 1))


def test_compute_p_matrix_sparse_matches_dense_on_observed_edges(uu_graph):
    cmvp = CMVP(uu_graph, seed=1)
    cmvp.fit_configuration_model()
    p_sparse = cmvp.compute_p_matrix_sparse()
    p_dense = cmvp.p_matrix
    coo = p_sparse.tocoo()
    np.testing.assert_allclose(coo.data, p_dense[coo.row, coo.col], rtol=1e-10)


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(uu_graph, tmp_path):
    cmvp = CMVP(uu_graph, seed=1)
    cmvp.fit_configuration_model()
    cmvp.validate_projection()
    cmvp.filter_backbone(alpha=0.5, correction='none')

    path = str(tmp_path / 'run')
    cmvp.save(path)
    loaded = CMVP.load(path)

    np.testing.assert_allclose(loaded.p_matrix, cmvp.p_matrix)
    np.testing.assert_allclose(loaded._pvalues_array, cmvp._pvalues_array)
    assert (loaded.backbone != cmvp.backbone).nnz == 0


# ---------------------------------------------------------------------------
# Input validation errors
# ---------------------------------------------------------------------------

def test_empty_graph_raises():
    with pytest.raises(ValueError):
        CMVP(nx.Graph())


def test_single_node_raises():
    G = nx.Graph()
    G.add_node(0)
    with pytest.raises(ValueError):
        CMVP(G)


def test_self_loop_raises():
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(0, 0)
    with pytest.raises(ValueError):
        CMVP(G)


def test_none_graph_raises():
    with pytest.raises(ValueError):
        CMVP(None)
