"""
Smoke tests for cmvp.utils validators: NullModelValidator and
BackboneValidator's main entry points (validate()) exercise most plot_*
methods internally, so these catch import/API breakage across the module
without individually testing every plot function.
"""

import matplotlib
matplotlib.use('Agg')

import pytest

from cmvp import CMVP
from cmvp.utils.null_model_validation import NullModelValidator
from cmvp.utils.backbone_validation import BackboneValidator


@pytest.fixture
def fitted_cmvp(uu_graph):
    cmvp = CMVP(uu_graph, seed=1)
    cmvp.fit_configuration_model()
    cmvp.validate_projection()
    cmvp.filter_backbone(alpha=0.5, correction='none')
    return cmvp


@pytest.fixture
def fitted_weighted_cmvp(uw_graph):
    cmvp = CMVP(uw_graph, seed=1)
    cmvp.fit_configuration_model()
    cmvp.validate_projection()
    cmvp.filter_backbone(alpha=0.5, correction='none')
    return cmvp


def test_null_model_validator_validate(fitted_cmvp):
    nv = NullModelValidator(fitted_cmvp)
    results = nv.validate(plot=True, verbose=False)
    assert 'degree' in results
    assert 'edge_probs' in results


def test_null_model_validator_validate_weighted(fitted_weighted_cmvp):
    nv = NullModelValidator(fitted_weighted_cmvp)
    results = nv.validate(plot=True, verbose=False)
    assert results['strength'] is not None


def test_null_model_validator_std_vs_expected_cn(fitted_cmvp):
    """Uses the shared ecdf_xy helper for its Fano-factor CDF panel."""
    nv = NullModelValidator(fitted_cmvp)
    nv.plot_std_vs_expected_cn(show=False)


def test_backbone_validator_validate(fitted_cmvp):
    bv = BackboneValidator(fitted_cmvp)
    results = bv.validate(plot=True, verbose=False)
    assert 'density' in results
    assert 'similarity' in results


def test_backbone_validator_sample_null_cn(fitted_cmvp):
    """Uses the shared sample_adjacency_from_p helper."""
    bv = BackboneValidator(fitted_cmvp)
    res = bv._sample_null_cn([(0, 1), (2, 3)], n_samples=10)
    assert len(res) == 2
    assert all(len(v) == 10 for v in res.values())


def test_backbone_validator_metric_correlation(fitted_cmvp):
    """Uses BackboneValidator._stat_matrix() for its z-score column."""
    bv = BackboneValidator(fitted_cmvp)
    bv.plot_metric_correlation(show=False)
