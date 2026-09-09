# CMVP — Configuration-Model Validated Projection

Statistical validation of network projections using configuration-model null hypotheses.

CMVP identifies statistically significant co-occurrences in monopartite network
projections. It fits a maximum-entropy configuration model (via
[NEMtropy](https://github.com/nicoloval/NEMtropy)) as a null model, derives
pair-specific edge probabilities, models the common-neighbor count as a
Poisson–Binomial random variable, and applies BH–FDR multiple-testing
correction to extract a validated backbone.

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.10. Dependencies (numpy, networkx, scipy, tqdm, matplotlib,
NEMtropy) are installed automatically.

## Quickstart

See also [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for a
runnable notebook version of this.

```python
from cmvp import CMVP
import networkx as nx

G = nx.karate_club_graph()

cmvp = CMVP(G, seed=42)
cmvp.fit_configuration_model(model='auto', method='fixed-point', max_iter=1000)
cmvp.validate_projection(test='poisson')
cmvp.filter_backbone(alpha=0.05, correction='fdr')

G_backbone = cmvp.to_networkx()
```

Or one-shot:

```python
from cmvp import cmvp_backbone

backbone, pvalues = cmvp_backbone(G, alpha=0.01)
```

## Command line

```bash
python -m cmvp your_network.graphml -o results/run --save
python -m cmvp results/run --load --filter-alpha 0.001
```

| Flag | Default | Notes |
|------|---------|-------|
| `input` | — | Graph file (`.graphml`, `.gml`, `.mtx`, `.edgelist`, `.json`, `.gexf`, `.npz`) or a CMVP save path (with `--load`) |
| `-o`, `--output` | — | Output path prefix (without extension) |
| `--save` | off | Save the full CMVP object for later re-filtering |
| `--load` | off | Load a saved CMVP object instead of a graph file |
| `--collapse-multi` | off | Collapse a multigraph to a simple graph |
| `--drop-weights` | off | Drop edge weights |
| `--model` | `auto` | NEMtropy model: `auto`, `cm_exp`, `cm`, `ecm_exp`, `ecm`, `crema`, `dcm_exp`, `dcm`, `decm_exp`, `decm` |
| `--method` | `quasinewton` | Solver method: `fixed-point`, `newton`, `quasinewton` |
| `--max-iter` | `5000` | Max solver iterations |
| `--seed` | — | Random seed |
| `--similarity` | `common_neighbors` | Similarity measure |
| `--test` | `poisson` | Statistical test: `poisson`, `normal`, `poisson-binomial` |
| `--tail` | `right` | Test tail: `right`, `left` |
| `--correction` | `fdr` | Multiple testing correction: `fdr`, `bonferroni`, `none` |
| `--alpha` | `0.05` | Significance level |
| `--target-density` | — | Target backbone density (overrides `--alpha`) |
| `--directed-mode` | `out-out` | Directed overlap mode |
| `--connected` | off | Filter for the minimum connected backbone |
| `--filter-alpha` | — | Re-filter with a different alpha (post-validation, no refit) |
| `--filter-density` | — | Re-filter with a different density (post-validation, no refit) |

## Running tests

```bash
pip install -e ".[test]"
pytest
```

## Core Workflow

### 1. Initialize CMVP

```python
from cmvp import CMVP
import networkx as nx

G = nx.read_graphml("your_network.graphml")
cmvp = CMVP(G, seed=42)
```

**Parameters:**
- `graph`: `networkx.Graph` or `networkx.DiGraph`
  Input network (can be directed/undirected, weighted/unweighted)
- `seed`: `int`, optional
  Random seed for reproducibility

---

### 2. Fit Configuration Model

```python
cmvp.fit_configuration_model(model='auto', method='fixed-point', max_iter=1000)
```

**Parameters:**

| Parameter | Type | Default | Options | Notes |
|-----------|------|---------|---------|-------|
| `model` | str | `'auto'` | `'auto'`, `'cm_exp'`, `'cm'`, `'ecm_exp'`, `'ecm'`, `'dcm_exp'`, `'dcm'`, `'decm_exp'`, `'decm'` | Auto-selects based on graph type (directed/weighted) |
| `method` | str | `'fixed-point'` | `'fixed-point'`, `'newton'`, `'quasinewtonian'` | Solver method. Note: DECM auto-switches to 'newton' (fixed-point poor) |
| `max_iter` | int | `1000` | Any positive int | Maximum solver iterations |
| `verbose` | bool | `True` | `True`, `False` | Print fitting progress |

**Auto Model Selection:**
```
Undirected, Unweighted  →  'cm_exp'       (Configuration Model, Exponential)
Undirected, Weighted    →  'ecm_exp'      (Enhanced Config Model, Exponential)
Directed, Unweighted    →  'dcm_exp'      (Directed Config Model, Exponential)
Directed, Weighted      →  'decm_exp'     (Directed Enhanced Config Model, Exponential)
```

---

### 3. Validate Projection

```python
cmvp.validate_projection(
    similarity='common_neighbors',
    directed_mode='out-out',
    tail='right',
    test='poisson',
    mid_p=False
)
```

**Parameters:**

| Parameter | Type | Default | Options | Notes |
|-----------|------|---------|---------|-------|
| `similarity` | str | `'common_neighbors'` | `'common_neighbors'` only | **Only common neighbors** supports statistical validation (analytical variance) |
| `directed_mode` | str | `'out-out'` | `'out-out'`, `'in-in'`, `'out-in'`, `'in-out'` | Overlap type for directed graphs |
| `tail` | str | `'right'` | `'right'`, `'left'`, `'both'` | `'right'`: test for similarity (observed >> expected). `'left'`: test for dissimilarity. `'both'`: signed test (requires `test='normal'` or `'poisson-binomial'`) |
| `test` | str | `'poisson'` | `'poisson'`, `'normal'`, `'poisson-binomial'` | Statistical test for p-values. Forced to `'normal'` for weighted null models |
| `mid_p` | bool | `False` | `True`, `False` | Mid-p correction for the discrete `'poisson'`/`'poisson-binomial'` tests (reduces conservativeness) |

This computes p-values only — call `filter_backbone()` (below) to apply
`alpha`/`correction`/`target_density` thresholds.

#### Test Types
```
'poisson'           : Poisson approximation (default, fast for sparse graphs)
'normal'            : Normal approximation with continuity correction (dense graphs)
'poisson-binomial'  : Exact Poisson-Binomial via DFT (most accurate, slowest)
```

---

### 4. Re-filter Backbone (without recomputation)

```python
cmvp.filter_backbone(alpha=0.05, correction='fdr')
# or
cmvp.filter_backbone(target_density=0.05)
# or
cmvp.filter_backbone(connected=True)
```

Change `alpha`, `correction`, or `target_density` and re-apply thresholds to
already-computed p-values without refitting the null model.

---

### 5. Convert to NetworkX

```python
G_backbone = cmvp.to_networkx(backbone=cmvp.backbone)
```

Returns a NetworkX graph with all original node attributes copied.

---

## Result Matrices

After `validate_projection()`, these are available:

| Attribute | Type | Contents |
|-----------|------|----------|
| `obs_sim_matrix` | ndarray | Observed similarity (common neighbors) |
| `exp_sim_matrix` | ndarray | Expected similarity under null model |
| `test_sim_matrix_pvalue` | sparse | Raw p-values (right-tail; also left-tail via `_pvalues_left` when `tail='both'`) |
| `jaccard_matrix` | ndarray | Jaccard similarity (computed separately for reference) |
| `backbone` | sparse CSR | Binary adjacency matrix of validated edges |
| `p_matrix` | ndarray | Edge probability matrix from null model |
| `w_matrix` | ndarray | Edge weight expectation matrix (weighted models) |

---

## Analysis Tools

### Null Model Validation

```python
from cmvp import NullModelValidator

nm = NullModelValidator(cmvp)
results = nm.validate(plot=True)
```

Checks degree preservation, strength preservation, and edge probability distribution.

### Backbone Validation

```python
from cmvp import BackboneValidator

bb = BackboneValidator(cmvp)
bb.plot_metric_correlation(method='rank', save_path="correlation.png")
```


## Folder Structure

```
cmvp/
├── cmvp/
│   ├── __init__.py       # public API: CMVP, cmvp_backbone, SIMILARITY_MEASURES, BackboneValidator, NullModelValidator
│   ├── core.py            # CMVP pipeline: fit null model → similarity → p-values → backbone
│   ├── cli.py              # command-line entry point (python -m cmvp)
│   └── utils/
│       ├── validation.py                    # shared plotting helpers (PlotConfig, _finalize_plot, ...)
│       ├── backbone_validation.py            # BackboneValidator base class
│       ├── backbone_validation_nullcal.py    # null-calibration diagnostics (mixin)
│       ├── backbone_validation_similarity.py # similarity-metric diagnostics (mixin)
│       └── null_model_validation.py          # NullModelValidator: degree/strength/probability checks
├── tests/
│   ├── conftest.py
│   ├── test_core_pipeline.py
│   ├── test_cli.py
│   └── test_utils.py
├── examples/
│   └── quickstart.ipynb  # runnable end-to-end example (karate club graph)
├── pyproject.toml
├── pytest.ini
├── LICENSE
└── README.md
```

## License

MIT — see [LICENSE](LICENSE).
