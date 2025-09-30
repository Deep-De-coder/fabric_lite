"""Model calibration utilities including temperature scaling and conformal prediction."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar


class TemperatureScaling:
    """Temperature scaling for model calibration."""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def fit_temperature(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Fit temperature parameter using validation set.
        
        Args:
            logits: Model logits [N, num_classes]
            labels: True labels [N]
        
        Returns:
            Optimal temperature value
        """
        def objective(T):
            scaled_logits = logits / T
            loss = nn.CrossEntropyLoss()(scaled_logits, labels)
            return loss.item()
        
        # Find optimal temperature
        result = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        
        return self.temperature
    
    def apply_temperature(self, logits: torch.Tensor, temperature: Optional[float] = None) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Model logits
            temperature: Temperature value (uses fitted value if None)
        
        Returns:
            Temperature-scaled logits
        """
        T = temperature if temperature is not None else self.temperature
        return logits / T
    
    def save(self, path: Path) -> None:
        """Save temperature value to file."""
        data = {"temperature": self.temperature}
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: Path) -> None:
        """Load temperature value from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.temperature = data["temperature"]


def conformal_thresholds(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    alpha: float = 0.1
) -> Dict[int, float]:
    """
    Compute conformal prediction thresholds for each class.
    
    Args:
        val_probs: Validation set probabilities [N, num_classes]
        val_labels: Validation set true labels [N]
        alpha: Significance level (1 - coverage)
    
    Returns:
        Dictionary mapping class index to threshold
    """
    num_classes = val_probs.shape[1]
    thresholds = {}
    
    for class_idx in range(num_classes):
        # Get probabilities for this class
        class_probs = val_probs[:, class_idx]
        
        # Get true labels for this class
        is_class = (val_labels == class_idx)
        
        if np.sum(is_class) == 0:
            # No examples of this class, set threshold to 0
            thresholds[class_idx] = 0.0
            continue
        
        # Get probabilities for true examples of this class
        true_class_probs = class_probs[is_class]
        
        # Compute threshold as (1-alpha) quantile
        threshold = np.quantile(true_class_probs, 1 - alpha)
        thresholds[class_idx] = threshold
    
    return thresholds


class ConformalPredictor:
    """Conformal prediction wrapper for classification."""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.thresholds: Optional[Dict[int, float]] = None
    
    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray) -> None:
        """Fit conformal prediction thresholds."""
        self.thresholds = conformal_thresholds(val_probs, val_labels, self.alpha)
    
    def predict(self, probs: np.ndarray) -> List[List[int]]:
        """
        Get conformal prediction sets.
        
        Args:
            probs: Prediction probabilities [N, num_classes]
        
        Returns:
            List of prediction sets for each sample
        """
        if self.thresholds is None:
            raise ValueError("Must fit conformal predictor before predicting")
        
        prediction_sets = []
        
        for sample_probs in probs:
            # Get classes that exceed their thresholds
            prediction_set = []
            for class_idx, threshold in self.thresholds.items():
                if sample_probs[class_idx] >= threshold:
                    prediction_set.append(class_idx)
            
            prediction_sets.append(prediction_set)
        
        return prediction_sets
    
    def save(self, path: Path) -> None:
        """Save conformal prediction thresholds."""
        if self.thresholds is None:
            raise ValueError("No thresholds to save")
        
        data = {
            "alpha": self.alpha,
            "thresholds": self.thresholds
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: Path) -> None:
        """Load conformal prediction thresholds."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.alpha = data["alpha"]
        self.thresholds = {int(k): v for k, v in data["thresholds"].items()}
