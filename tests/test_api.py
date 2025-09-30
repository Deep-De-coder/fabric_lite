"""Tests for FabricLite API."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from fabriclite import FabricClassifier, FABRIC_LABELS


def create_test_image(size=(224, 224), color=(128, 128, 128)):
    """Create a test image."""
    return Image.new('RGB', size, color)


def test_classifier_initialization():
    """Test classifier initialization."""
    classifier = FabricClassifier.from_pretrained()
    
    assert classifier is not None
    assert classifier.device in ['cpu', 'cuda']
    assert len(classifier.labels) == len(FABRIC_LABELS)
    assert classifier.temperature == 1.0


def test_predict_with_pil_image():
    """Test prediction with PIL Image."""
    classifier = FabricClassifier.from_pretrained()
    
    # Create test image
    test_image = create_test_image()
    
    # Make prediction
    result = classifier.predict(test_image)
    
    # Check result format
    assert isinstance(result, dict)
    assert len(result) <= len(FABRIC_LABELS)
    
    # Check that all keys are valid labels
    for label in result.keys():
        assert label in FABRIC_LABELS
    
    # Check that probabilities are valid (between 0 and 1)
    for prob in result.values():
        assert 0.0 <= prob <= 1.0
    
    # Check that we have at least one prediction
    assert len(result) > 0


def test_predict_with_file_path():
    """Test prediction with file path."""
    classifier = FabricClassifier.from_pretrained()
    
    # Create temporary image file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        test_image = create_test_image()
        test_image.save(tmp.name, 'JPEG')
        
        try:
            # Make prediction
            result = classifier.predict(tmp.name)
            
            # Check result format
            assert isinstance(result, dict)
            assert len(result) <= len(FABRIC_LABELS)
            
            # Check that probabilities are valid (between 0 and 1)
            for prob in result.values():
                assert 0.0 <= prob <= 1.0
            
            # Check that we have at least one prediction
            assert len(result) > 0
            
        finally:
            # Clean up
            Path(tmp.name).unlink()


def test_predict_with_tensor():
    """Test prediction with tensor input."""
    classifier = FabricClassifier.from_pretrained()
    
    # Create test tensor
    test_tensor = torch.randn(1, 3, 224, 224)
    
    # Make prediction
    result = classifier.predict(test_tensor)
    
    # Check result format
    assert isinstance(result, dict)
    assert len(result) <= len(FABRIC_LABELS)


def test_predict_batch():
    """Test batch prediction."""
    classifier = FabricClassifier.from_pretrained()
    
    # Create test images
    test_images = [create_test_image() for _ in range(3)]
    
    # Make batch prediction
    results = classifier.predict_batch(test_images)
    
    # Check result format
    assert isinstance(results, list)
    assert len(results) == 3
    
    for result in results:
        assert isinstance(result, dict)
        assert len(result) <= len(FABRIC_LABELS)


def test_predict_proba():
    """Test probability prediction."""
    classifier = FabricClassifier.from_pretrained()
    
    # Create test image
    test_image = create_test_image()
    
    # Get probabilities
    probs = classifier.predict_proba(test_image)
    
    # Check result format
    assert isinstance(probs, torch.Tensor)
    assert probs.shape == (len(FABRIC_LABELS),)
    
    # Check that probabilities sum to approximately 1
    prob_sum = probs.sum().item()
    assert abs(prob_sum - 1.0) < 0.01


def test_temperature_scaling():
    """Test temperature scaling."""
    classifier = FabricClassifier.from_pretrained()
    
    # Test temperature setting
    classifier.temperature = 2.0
    assert classifier.temperature == 2.0
    
    # Test calibration with dummy data
    dummy_logits = torch.randn(10, len(FABRIC_LABELS))
    dummy_labels = torch.randint(0, len(FABRIC_LABELS), (10,))
    
    optimal_temp = classifier.calibrate(dummy_logits, dummy_labels)
    assert isinstance(optimal_temp, float)
    assert optimal_temp > 0


def test_white_balance():
    """Test white balance correction."""
    classifier = FabricClassifier.from_pretrained()
    
    # Create test image
    test_image = create_test_image()
    
    # Test with and without white balance
    result_no_wb = classifier.predict(test_image, white_balance=False)
    result_with_wb = classifier.predict(test_image, white_balance=True)
    
    # Both should return valid results
    assert isinstance(result_no_wb, dict)
    assert isinstance(result_with_wb, dict)
    
    # Probabilities should be valid
    for prob in result_no_wb.values():
        assert 0.0 <= prob <= 1.0
    for prob in result_with_wb.values():
        assert 0.0 <= prob <= 1.0


def test_topk_parameter():
    """Test topk parameter."""
    classifier = FabricClassifier.from_pretrained()
    
    # Create test image
    test_image = create_test_image()
    
    # Test different topk values
    for topk in [1, 3, 5, len(FABRIC_LABELS)]:
        result = classifier.predict(test_image, topk=topk)
        assert len(result) <= topk
        assert len(result) <= len(FABRIC_LABELS)


def test_model_save_load():
    """Test model saving and loading."""
    classifier = FabricClassifier.from_pretrained()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        try:
            # Save model
            classifier.save(tmp.name)
            
            # Check that files were created
            assert Path(tmp.name).exists()
            assert Path(tmp.name.replace('.pt', '_calibration.json')).exists()
            
        finally:
            # Clean up
            Path(tmp.name).unlink(missing_ok=True)
            Path(tmp.name.replace('.pt', '_calibration.json')).unlink(missing_ok=True)
