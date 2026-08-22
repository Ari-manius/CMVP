"""
CLI regression tests. Covers the bug where the default pipeline path never
called filter_backbone() (args.alpha was parsed but unused, so to_networkx()
always raised) — see cli.py run_pipeline().
"""

import argparse
from pathlib import Path

import networkx as nx
import pytest

from cmvp import cli


def _default_args(**overrides):
    defaults = dict(
        input=None, output=None, save=False, load=False,
        collapse_multi=False, drop_weights=False,
        model='auto', method='fixed-point', max_iter=5000, seed=1,
        similarity='common_neighbors', test='poisson', tail='right',
        correction='none', alpha=0.5, target_density=None, directed_mode='out-out',
        connected=False, filter_alpha=None, filter_density=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def graphml_path(tmp_path, uu_graph):
    path = tmp_path / 'graph.graphml'
    nx.write_graphml(uu_graph, str(path))
    return str(path)


def test_run_pipeline_default_alpha_produces_backbone(graphml_path):
    """Regression test: default path (no --connected/--filter-*) must call
    filter_backbone() using args.alpha, not leave the backbone unset."""
    args = _default_args(input=graphml_path)
    cmvp, G_backbone = cli.run_pipeline(args)
    assert cmvp.backbone is not None
    assert set(G_backbone.nodes())


def test_run_pipeline_connected(graphml_path):
    args = _default_args(input=graphml_path, connected=True)
    cmvp, G_backbone = cli.run_pipeline(args)
    assert nx.is_connected(G_backbone)


def test_run_pipeline_filter_density(graphml_path):
    args = _default_args(input=graphml_path, filter_density=0.3)
    cmvp, G_backbone = cli.run_pipeline(args)
    assert cmvp.backbone is not None


def test_run_pipeline_save_and_load(graphml_path, tmp_path):
    out_prefix = str(tmp_path / 'run')
    args = _default_args(input=graphml_path, output=out_prefix, save=True)
    cmvp, G_backbone = cli.run_pipeline(args)
    assert Path(out_prefix + '.npz').exists()
    assert Path(out_prefix + '_backbone.graphml').exists()

    from cmvp.core import CMVP
    loaded = CMVP.load(out_prefix)
    assert (loaded.backbone != cmvp.backbone).nnz == 0


def test_load_graph_unsupported_extension(tmp_path):
    bogus = tmp_path / 'graph.xyz'
    bogus.write_text('nonsense')
    with pytest.raises(ValueError):
        cli.load_graph(str(bogus))


def test_load_graph_missing_file():
    with pytest.raises(FileNotFoundError):
        cli.load_graph('/nonexistent/path/graph.graphml')


def test_preprocess_removes_selfloops_and_isolates():
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 1)  # self-loop
    G.add_node(2)      # isolate
    G_clean = cli.preprocess_graph(G)
    assert G_clean.number_of_nodes() == 2
    assert nx.number_of_selfloops(G_clean) == 0
