"""Constants for FabricLite package."""

# Canonical class names in stable order
CLASS_NAMES = [
    "cotton", "denim", "leather", "linen",
    "silk", "synthetic", "velvet", "wool",
]

# CSV header for structured output
CSV_HEADER = ["image", "predicted_label", "confidence"] + CLASS_NAMES
