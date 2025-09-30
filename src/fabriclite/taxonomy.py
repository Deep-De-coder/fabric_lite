"""Fabric taxonomy and label definitions."""

from typing import Dict, List

# Canonical fabric labels
FABRIC_LABELS: List[str] = [
    "cotton",
    "denim", 
    "leather",
    "silk",
    "velvet",
    "wool",
    "linen",
    "synthetic"
]

# Label synonyms for flexible matching
SYNONYMS: Dict[str, List[str]] = {
    "denim": ["jeans fabric", "blue denim", "denim twill", "jeans"],
    "synthetic": ["polyester", "nylon", "rayon", "viscose", "acrylic", "poly", "plastic"],
    "leather": ["genuine leather", "faux leather", "pleather", "suede", "hide"],
    "cotton": ["cotton blend", "organic cotton", "cotton twill"],
    "silk": ["silk blend", "satin", "chiffon", "crepe"],
    "velvet": ["velour", "velveteen", "corduroy"],
    "wool": ["merino", "cashmere", "alpaca", "sheep wool"],
    "linen": ["flax", "linen blend", "hemp"]
}

# Create label to index mapping
LABEL_TO_IDX: Dict[str, int] = {label: idx for idx, label in enumerate(FABRIC_LABELS)}
IDX_TO_LABEL: Dict[int, str] = {idx: label for label, idx in LABEL_TO_IDX.items()}

def normalize_label(label: str) -> str:
    """Normalize a label by checking synonyms and returning canonical form."""
    label_lower = label.lower().strip()
    
    # Direct match
    if label_lower in FABRIC_LABELS:
        return label_lower
    
    # Check synonyms
    for canonical, synonyms in SYNONYMS.items():
        if label_lower in [s.lower() for s in synonyms]:
            return canonical
    
    # Return original if no match found
    return label_lower
