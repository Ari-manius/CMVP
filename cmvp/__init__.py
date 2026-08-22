"""
Configuration-Model Validated Projection (CMVP)
================================================

Statistical backbone extraction for monopartite networks.

Supports:
- Directed and undirected graphs
- Weighted and unweighted (binary) edges
- Common neighbors similarity (primary); Jaccard available for comparison
- Multiple NEMtropy null models (UBCM, UECM, DBCM, etc.)

Main Classes and Functions:
    CMVP: Main class for backbone extraction
    cmvp_backbone: One-shot function for quick backbone extraction

Example Usage:
    >>> from cmvp import CMVP, cmvp_backbone
    >>> import networkx as nx
    >>>
    >>> # Load your network
    >>> G = nx.karate_club_graph()
    >>>
    >>> # Quick extraction
    >>> backbone, pvalues = cmvp_backbone(G, alpha=0.01)
    >>>
    >>> # Or use the class for more control
    >>> cmvp = CMVP(G)
    >>> cmvp.fit_configuration_model()
    >>> backbone, pvalues = cmvp.validate_projection(alpha=0.01)
    >>> G_backbone = cmvp.to_networkx(backbone, weighted=True, pvalues=pvalues)
"""

__version__ = '1.0.0'
__author__ = 'CMVP Team'

from .core import CMVP, cmvp_backbone, SIMILARITY_MEASURES

from .utils.backbone_validation import BackboneValidator
from .utils.null_model_validation import NullModelValidator

__all__ = [
    'CMVP',
    'cmvp_backbone',
    'SIMILARITY_MEASURES',
    'BackboneValidator',
    'NullModelValidator',
]
