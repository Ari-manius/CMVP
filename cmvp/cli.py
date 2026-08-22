"""
CMVP Command-Line Interface
============================

Run the full CMVP pipeline from the terminal.

Usage:
    python -m cmvp input.graphml -o results/my_run
    python -m cmvp input.mtx --model cm_exp --alpha 0.01
    python -m cmvp input.graphml --connected --save
"""

import argparse
import sys
import os
import time
import networkx as nx
import numpy as np
from pathlib import Path


def load_graph(path: str) -> nx.Graph:
    """
    Load a graph from file. Supports graphml, gml, edgelist, adjlist, mtx.
    Auto-detects format from extension.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    ext = path.suffix.lower()

    if ext == '.graphml':
        G = nx.read_graphml(str(path))
    elif ext == '.gml':
        G = nx.read_gml(str(path))
    elif ext in ('.edgelist', '.edges', '.txt', '.csv'):
        G = nx.read_edgelist(str(path))
    elif ext == '.adjlist':
        G = nx.read_adjlist(str(path))
    elif ext == '.mtx':
        from scipy.io import mmread
        a = mmread(str(path))
        G = nx.Graph(a)
    elif ext == '.npz':
        from scipy import sparse
        a = sparse.load_npz(str(path))
        G = nx.Graph(a)
    elif ext in ('.gexf',):
        G = nx.read_gexf(str(path))
    elif ext == '.json':
        import json
        with open(path) as f:
            data = json.load(f)
        G = nx.node_link_graph(data)
    else:
        raise ValueError(f"Unsupported format: {ext}. "
                         f"Supported: .graphml, .gml, .gexf, .mtx, .npz, .edgelist, .txt, .csv, .adjlist, .json")

    return G


def preprocess_graph(G: nx.Graph, collapse_multi: bool = False,
                     drop_weights: bool = False) -> nx.Graph:
    """Clean graph: remove self-loops, isolates, optionally collapse multigraph."""
    n_before = G.number_of_nodes()
    e_before = G.number_of_edges()

    # Collapse multigraph if needed
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        if collapse_multi:
            if G.is_directed():
                G_new = nx.DiGraph()
            else:
                G_new = nx.Graph()
            for u, v, data in G.edges(data=True):
                if G_new.has_edge(u, v):
                    G_new[u][v]['weight'] = G_new[u][v].get('weight', 1) + data.get('weight', 1)
                else:
                    G_new.add_edge(u, v, weight=data.get('weight', 1))
            for node in G.nodes():
                G_new.add_node(node, **dict(G.nodes[node]))
            G = G_new
            print(f"  Collapsed multigraph to simple graph")
        else:
            raise ValueError("Input is a multigraph. Use --collapse-multi to collapse it.")

    # Remove self-loops
    selfloops = list(nx.selfloop_edges(G))
    if selfloops:
        G.remove_edges_from(selfloops)
        print(f"  Removed {len(selfloops)} self-loops")

    # Remove isolates
    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)
        print(f"  Removed {len(isolates)} isolated nodes")

    if drop_weights:
        for u, v in G.edges():
            G[u][v].clear()
        print(f"  Dropped edge weights")

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"(was {n_before}/{e_before})")
    print(f"  Directed: {G.is_directed()}, Weighted: {nx.is_weighted(G)}")

    return G


def run_pipeline(args):
    """Run the full CMVP pipeline."""
    from .core import CMVP

    t_start = time.time()

    # --- Load ---
    print(f"\n{'='*60}")
    print(f"CMVP Pipeline")
    print(f"{'='*60}")
    print(f"\nLoading graph: {args.input}")
    G = load_graph(args.input)

    # --- Preprocess ---
    print("\nPreprocessing...")
    G = preprocess_graph(G, collapse_multi=args.collapse_multi,
                         drop_weights=args.drop_weights)

    # --- Fit ---
    print(f"\nFitting null model...")
    cmvp = CMVP(G, seed=args.seed)
    cmvp.fit_configuration_model(
        model=args.model,
        method=args.method,
        max_iter=args.max_iter
    )

    # --- Validate ---
    print(f"\nValidating projection...")
    cmvp.validate_projection(
        similarity=args.similarity,
        test=args.test,
        tail=args.tail,
        directed_mode=args.directed_mode,
    )

    # --- Filter ---
    if args.connected:
        print(f"\nFiltering for connected backbone...")
        cmvp.filter_backbone(connected=True)
    elif args.filter_alpha is not None:
        print(f"\nRe-filtering with alpha={args.filter_alpha}...")
        cmvp.filter_backbone(alpha=args.filter_alpha, correction=args.correction)
    elif args.filter_density is not None:
        print(f"\nRe-filtering with target_density={args.filter_density}...")
        cmvp.filter_backbone(target_density=args.filter_density)
    else:
        cmvp.filter_backbone(alpha=args.alpha, correction=args.correction)

    # --- Export ---
    G_backbone = cmvp.to_networkx()

    elapsed = time.time() - t_start
    print(f"\nBackbone: {G_backbone.number_of_nodes()} nodes, {G_backbone.number_of_edges()} edges")
    print(f"Density: {nx.density(G_backbone):.4f}")
    print(f"Time: {elapsed:.1f}s")

    # --- Save ---
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)

        if args.save:
            cmvp.save(str(out))

        # Always export backbone graph
        backbone_path = str(out) + "_backbone.graphml"
        nx.write_graphml(G_backbone, backbone_path)
        print(f"\nSaved backbone to {backbone_path}")

        if args.save:
            print(f"Saved CMVP object to {out} (.npz, _graph.graphml, _meta.json)")
    elif args.save:
        print("\nWarning: --save requires --output / -o")

    print(f"\nDone.")
    return cmvp, G_backbone


def main():
    parser = argparse.ArgumentParser(
        prog='cmvp',
        description='CMVP - Configuration Model Validated Projection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cmvp karate.graphml -o results/karate
  python -m cmvp data.mtx --model cm_exp --alpha 0.01 -o results/run1 --save
  python -m cmvp net.graphml --connected -o results/connected
  python -m cmvp net.graphml --target-density 0.2 -o results/dense
  python -m cmvp saved_run -o results/refiltered --load --filter-alpha 0.001
        """
    )

    # Input
    parser.add_argument('input', help='Path to graph file (.graphml, .gml, .mtx, .edgelist, .json, .gexf, .npz) or CMVP save path (with --load)')

    # Output
    parser.add_argument('-o', '--output', help='Output path prefix (without extension)')
    parser.add_argument('--save', action='store_true', help='Save full CMVP object for later re-filtering')

    # Load mode
    parser.add_argument('--load', action='store_true', help='Load a saved CMVP object instead of a graph file')

    # Preprocessing
    parser.add_argument('--collapse-multi', action='store_true', help='Collapse multigraph to simple graph')
    parser.add_argument('--drop-weights', action='store_true', help='Drop edge weights')

    # Null model
    parser.add_argument('--model', default='auto', help='NEMtropy model (default: auto). Options: auto, cm_exp, cm, ecm_exp, ecm, crema, dcm_exp, dcm, decm_exp, decm')
    parser.add_argument('--method', default='quasinewton', help='Solver method (default: quasinewton). Options: fixed-point, newton, quasinewton')
    parser.add_argument('--max-iter', type=int, default=5000, help='Max solver iterations (default: 5000)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')

    # Validation
    parser.add_argument('--similarity', default='common_neighbors', help='Similarity measure (default: common_neighbors)')
    parser.add_argument('--test', default='poisson', help='Statistical test (default: poisson). Options: poisson, normal, poisson-binomial')
    parser.add_argument('--tail', default='right', help='Test tail (default: right). Options: right, left')
    parser.add_argument('--correction', default='fdr', help='Multiple testing correction (default: fdr). Options: fdr, bonferroni, none')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level (default: 0.05)')
    parser.add_argument('--target-density', type=float, default=None, help='Target backbone density (overrides alpha)')
    parser.add_argument('--directed-mode', default='out-out', help='Directed overlap mode (default: out-out)')

    # Filtering (applied after validation)
    parser.add_argument('--connected', action='store_true', help='Filter for minimum connected backbone')
    parser.add_argument('--filter-alpha', type=float, default=None, help='Re-filter with different alpha (post-validation)')
    parser.add_argument('--filter-density', type=float, default=None, help='Re-filter with different density (post-validation)')

    # Export

    args = parser.parse_args()

    # Load mode: reload saved CMVP and re-filter
    if args.load:
        from .core import CMVP
        print(f"\nLoading saved CMVP from: {args.input}")
        cmvp = CMVP.load(args.input)

        if args.connected:
            cmvp.filter_backbone(connected=True)
        elif args.filter_alpha is not None:
            cmvp.filter_backbone(alpha=args.filter_alpha, correction=args.correction)
        elif args.filter_density is not None:
            cmvp.filter_backbone(target_density=args.filter_density)

        G_backbone = cmvp.to_networkx()
        print(f"\nBackbone: {G_backbone.number_of_nodes()} nodes, {G_backbone.number_of_edges()} edges")

        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            backbone_path = str(out) + "_backbone.graphml"
            nx.write_graphml(G_backbone, backbone_path)
            print(f"Saved backbone to {backbone_path}")
            if args.save:
                cmvp.save(str(out))
    else:
        run_pipeline(args)


if __name__ == '__main__':
    main()
