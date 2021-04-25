"""
smote_likes implements SMOTE-like algorithms for synthetical data generation
"""

from ._version import __version__
from . import distance_metrics
from . import synth_generator

__all__ = [
    'distance_metrics',
    'synth_generator'
]