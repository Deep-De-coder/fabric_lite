# FabricLite

A tiny fabric/material classifier for garments with calibrated mixture outputs and mobile-friendly exports.

## Features

- **8 Fabric Types**: cotton, denim, leather, silk, velvet, wool, linen, synthetic
- **MobileNetV3-Small Backbone**: Efficient and accurate classification
- **Calibrated Outputs**: Temperature scaling and conformal prediction
- **Multiple Export Formats**: ONNX, TorchScript (TFLite stub)
- **CLI Interface**: Complete command-line tooling
- **FastAPI Server**: Production-ready microservice
- **White Balance Correction**: Optional gray-world preprocessing

## Installation

```bash
pip install fabriclite
```

## Quickstart

### Python API

```python
from fabriclite import FabricClassifier

# Load pretrained model
classifier = FabricClassifier.from_pretrained()

# Classify an image
result = classifier.predict("path/to/image.jpg")
print(result)
# {'cotton': 0.82, 'denim': 0.12, 'silk': 0.06}

# Batch processing
results = classifier.predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])

# With white balance correction
result = classifier.predict("image.jpg", white_balance=True)
```

### CLI Usage

```bash
# Single image inference
fabriclite infer image.jpg

# Batch processing
fabriclite batch /path/to/images --output results.csv

# Train a model
fabriclite train /path/to/train /path/to/val --epochs 15

# Evaluate model
fabriclite eval /path/to/test weights.pt

# Calibrate model
fabriclite calibrate /path/to/val weights.pt --output temp.json

# Export model
fabriclite export weights.pt --format onnx --output model.onnx
```

### FastAPI Server

```bash
# Start server
python examples/server_fastapi.py

# Or with uvicorn
uvicorn examples.server_fastapi:app --reload
```

Visit `http://localhost:8000/docs` for API documentation.

## Model Export

### ONNX Export

```python
from fabriclite.export import to_onnx

# Export to ONNX
to_onnx(classifier.model, "model.onnx")
```

### TorchScript Export

```python
from fabriclite.export import to_torchscript

# Export to TorchScript
to_torchscript(classifier.model, "model.pt")
```

## Calibration

FabricLite supports temperature scaling for better calibration:

```python
# Calibrate model
optimal_temp = classifier.calibrate(val_logits, val_labels)
print(f"Optimal temperature: {optimal_temp}")

# Use calibrated predictions
result = classifier.predict("image.jpg")
```

## Dataset Preparation

Organize your data in the following structure:

```
data/
├── train/
│   ├── cotton/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── denim/
│   │   └── image3.jpg
│   └── ...
├── val/
│   ├── cotton/
│   └── ...
└── test/
    ├── cotton/
    └── ...
```

## Training

```bash
# Basic training
fabriclite train data/train data/val --epochs 15 --lr 3e-4

# With white balance
fabriclite train data/train data/val --wb --epochs 20

# Custom batch size
fabriclite train data/train data/val --batch-size 32
```

Training will save:
- `artifacts/weights.pt` - Best model weights
- `artifacts/metadata.json` - Training metadata
- `artifacts/training_history.png` - Training curves

## Evaluation

```bash
# Evaluate model
fabriclite eval data/test weights.pt --report report.json --cm confusion.png
```

This generates:
- Accuracy and F1 scores
- Confusion matrix plot
- Detailed classification report

## API Reference

### FabricClassifier

```python
class FabricClassifier:
    @classmethod
    def from_pretrained(cls, name="mobilenet_v3_small", device=None, weights_url=None)
    
    def predict(self, x, topk=3, white_balance=False) -> Dict[str, float]
    def predict_batch(self, images, topk=3, white_balance=False) -> List[Dict[str, float]]
    def predict_proba(self, x, white_balance=False) -> torch.Tensor
    def calibrate(self, val_logits, val_labels) -> float
    def save(self, path)
    def load_calibration(self, path)
```

### Preprocessing

```python
from fabriclite.preprocess import preprocess, apply_gray_world

# Basic preprocessing
tensor = preprocess("image.jpg")

# With white balance
tensor = preprocess("image.jpg", white_balance=True)

# Custom size
tensor = preprocess("image.jpg", size=128)
```

## Available Fabric Types

- **cotton** - Cotton and cotton blends
- **denim** - Denim, jeans fabric, blue denim
- **leather** - Genuine leather, faux leather, suede
- **silk** - Silk, satin, chiffon, crepe
- **velvet** - Velvet, velour, velveteen
- **wool** - Wool, merino, cashmere, alpaca
- **linen** - Linen, flax, hemp
- **synthetic** - Polyester, nylon, rayon, viscose, acrylic

## Model Weights

Pretrained weights are automatically downloaded from Hugging Face Hub. You can also:

1. Set `FABRICLITE_WEIGHTS` environment variable to point to local weights
2. Use `--weights` parameter in CLI commands
3. Provide custom `weights_url` in `from_pretrained()`

## Development

```bash
# Clone repository
git clone https://github.com/Deep-De-coder/fabric_lite.git
cd fabric_lite

# Install in development mode
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
```

## Makefile Targets

```bash
make setup      # Install dependencies and setup pre-commit
make test       # Run tests
make fmt        # Format code
make lint       # Lint code
make serve      # Start FastAPI server
make clean      # Clean build artifacts
```

## License

Apache-2.0 License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Citation

```bibtex
@software{fabriclite2024,
  title={FabricLite: A Tiny Fabric Classifier for Garments},
  author={FabricLite Team},
  year={2024},
  url={https://github.com/Deep-De-coder/fabric_lite}
}
```

## Changelog

### v0.1.0
- Initial release
- MobileNetV3-Small backbone
- 8 fabric type classification
- CLI and FastAPI interfaces
- ONNX and TorchScript export
- Temperature scaling calibration
- White balance preprocessing
