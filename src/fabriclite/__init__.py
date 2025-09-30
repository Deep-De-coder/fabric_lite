"""FabricLite: A tiny fabric/material classifier for garments."""

from .classifier import FabricClassifier
from .taxonomy import FABRIC_LABELS, SYNONYMS

__version__ = "0.1.0"
__all__ = ["FabricClassifier", "FABRIC_LABELS", "SYNONYMS"]
