"""Main FabricClassifier class for fabric classification."""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from .calibration import TemperatureScaling
from .constants import CLASS_NAMES
from .models import create_model
from .preprocess import preprocess, preprocess_batch
from .taxonomy import FABRIC_LABELS, IDX_TO_LABEL
from .utils import get_model_weights_path, setup_logger


class FabricClassifier:
    """Main fabric classifier class."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        temperature: float = 1.0
    ):
        """
        Initialize classifier.
        
        Args:
            model: PyTorch model
            device: Device to run on ('cpu', 'cuda', or None for auto)
            temperature: Temperature for calibration
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.temperature_scaling = TemperatureScaling(temperature)
        self.labels = FABRIC_LABELS
        self.logger = setup_logger()
    
    @classmethod
    def from_pretrained(
        cls,
        name: str = "mobilenet_v3_small",
        device: Optional[str] = None,
        weights_url: Optional[str] = None
    ) -> "FabricClassifier":
        """
        Load pretrained classifier.
        
        Args:
            name: Model name
            device: Device to run on
            weights_url: Custom weights URL
        
        Returns:
            FabricClassifier instance
        """
        logger = setup_logger()
        
        # Create model
        model = create_model(num_classes=len(FABRIC_LABELS), pretrained_backbone=True)
        
        # Try to load weights
        weights_path = get_model_weights_path(name, weights_url)
        
        if weights_path and weights_path.exists():
            try:
                state_dict = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(state_dict)
                logger.info(f"Loaded pretrained weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Failed to load weights from {weights_path}: {e}")
                logger.warning("Using randomly initialized head")
        else:
            logger.warning("No pretrained weights found, using randomly initialized head")
            logger.warning("Model will still work but may have lower accuracy")
        
        return cls(model, device)
    
    def predict(
        self,
        x: Union[str, Path, bytes, torch.Tensor],
        topk: int = 3,
        white_balance: bool = False
    ) -> Dict[str, float]:
        """
        Predict fabric type for single image.
        
        Args:
            x: Input image (path, bytes, or tensor)
            topk: Number of top predictions to return
            white_balance: Whether to apply white balance correction
        
        Returns:
            Dictionary mapping labels to probabilities
        """
        # Preprocess input
        input_tensor = preprocess(x, white_balance=white_balance)
        
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(input_tensor)
            
            # Apply temperature scaling
            scaled_logits = self.temperature_scaling.apply_temperature(logits)
            
            # Get probabilities
            probs = F.softmax(scaled_logits, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, min(topk, len(self.labels)), dim=1)
            
            # Convert to dictionary
            result = {}
            for i in range(top_indices.shape[1]):
                label_idx = top_indices[0, i].item()
                prob = top_probs[0, i].item()
                label = IDX_TO_LABEL[label_idx]
                result[label] = prob
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, Path, bytes, torch.Tensor]],
        topk: int = 3,
        white_balance: bool = False
    ) -> List[Dict[str, float]]:
        """
        Predict fabric types for batch of images.
        
        Args:
            images: List of input images
            topk: Number of top predictions to return
            white_balance: Whether to apply white balance correction
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for img in images:
            result = self.predict(img, topk=topk, white_balance=white_balance)
            results.append(result)
        
        return results
    
    def predict_proba(
        self,
        x: Union[str, Path, bytes, torch.Tensor],
        white_balance: bool = False
    ) -> Dict[str, float]:
        """
        Get probability distribution as a dictionary.
        
        Args:
            x: Input image
            white_balance: Whether to apply white balance correction
        
        Returns:
            Dictionary mapping class names to probabilities
        """
        # Preprocess input
        input_tensor = preprocess(x, white_balance=white_balance)
        
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(input_tensor)
            scaled_logits = self.temperature_scaling.apply_temperature(logits)
            probs = F.softmax(scaled_logits, dim=1)
        
        # Convert to dictionary using canonical class names
        prob_dict = {}
        for i, class_name in enumerate(CLASS_NAMES):
            if i < probs.shape[1]:
                prob_dict[class_name] = float(probs[0, i].item())
            else:
                prob_dict[class_name] = 0.0
        
        return prob_dict
    
    def calibrate(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor
    ) -> float:
        """
        Calibrate model using temperature scaling.
        
        Args:
            val_logits: Validation set logits
            val_labels: Validation set labels
        
        Returns:
            Optimal temperature value
        """
        return self.temperature_scaling.fit_temperature(val_logits, val_labels)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model and calibration parameters."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        torch.save(self.model.state_dict(), path)
        
        # Save calibration parameters
        calib_path = path.parent / f"{path.stem}_calibration.json"
        self.temperature_scaling.save(calib_path)
        
        self.logger.info(f"Model saved to {path}")
        self.logger.info(f"Calibration saved to {calib_path}")
    
    def load_calibration(self, path: Union[str, Path]) -> None:
        """Load calibration parameters."""
        path = Path(path)
        self.temperature_scaling.load(path)
        self.logger.info(f"Calibration loaded from {path}")
    
    @property
    def temperature(self) -> float:
        """Get current temperature value."""
        return self.temperature_scaling.temperature
    
    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set temperature value."""
        self.temperature_scaling.temperature = value
