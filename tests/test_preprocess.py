"""Tests for preprocessing utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from fabriclite.preprocess import (
    load_image,
    apply_gray_world,
    preprocess,
    preprocess_batch
)


def test_load_image_from_path():
    """Test loading image from file path."""
    # Create temporary image file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        try:
            # Create and save test image
            test_image = Image.new('RGB', (100, 100), (255, 0, 0))
            test_image.save(tmp.name, 'JPEG')
            
            # Load image
            loaded_image = load_image(tmp.name)
            
            # Check result
            assert isinstance(loaded_image, Image.Image)
            assert loaded_image.mode == 'RGB'
            assert loaded_image.size == (100, 100)
            
        finally:
            # Clean up
            Path(tmp.name).unlink(missing_ok=True)


def test_load_image_from_pil():
    """Test loading image from PIL Image."""
    # Create test image
    test_image = Image.new('RGB', (50, 50), (0, 255, 0))
    
    # Load image
    loaded_image = load_image(test_image)
    
    # Check result
    assert isinstance(loaded_image, Image.Image)
    assert loaded_image.mode == 'RGB'
    assert loaded_image.size == (50, 50)


def test_load_image_from_numpy():
    """Test loading image from numpy array."""
    # Create test array
    test_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    # Load image
    loaded_image = load_image(test_array)
    
    # Check result
    assert isinstance(loaded_image, Image.Image)
    assert loaded_image.mode == 'RGB'
    assert loaded_image.size == (100, 100)


def test_load_image_from_bytes():
    """Test loading image from bytes."""
    # Create test image and convert to bytes
    test_image = Image.new('RGB', (75, 75), (0, 0, 255))
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        try:
            test_image.save(tmp.name, 'JPEG')
            
            # Read bytes
            with open(tmp.name, 'rb') as f:
                image_bytes = f.read()
            
            # Load image from bytes
            loaded_image = load_image(image_bytes)
            
            # Check result
            assert isinstance(loaded_image, Image.Image)
            assert loaded_image.mode == 'RGB'
            
        finally:
            # Clean up
            Path(tmp.name).unlink(missing_ok=True)


def test_load_image_invalid_type():
    """Test loading image with invalid type."""
    with pytest.raises(ValueError):
        load_image(123)  # Invalid type


def test_apply_gray_world():
    """Test gray world white balance correction."""
    # Create test image with color cast
    test_array = np.array([
        [[200, 100, 100], [200, 100, 100]],
        [[200, 100, 100], [200, 100, 100]]
    ], dtype=np.uint8)
    
    # Apply gray world correction
    corrected = apply_gray_world(test_array)
    
    # Check result
    assert isinstance(corrected, np.ndarray)
    assert corrected.shape == test_array.shape
    assert corrected.dtype == np.uint8
    
    # Check that values are in valid range
    assert corrected.min() >= 0
    assert corrected.max() <= 255


def test_preprocess_basic():
    """Test basic preprocessing."""
    # Create test image
    test_image = Image.new('RGB', (300, 300), (128, 128, 128))
    
    # Preprocess
    tensor = preprocess(test_image)
    
    # Check result
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224
    assert tensor.dtype == torch.float32


def test_preprocess_with_white_balance():
    """Test preprocessing with white balance."""
    # Create test image
    test_image = Image.new('RGB', (300, 300), (200, 100, 100))
    
    # Preprocess with white balance
    tensor = preprocess(test_image, white_balance=True)
    
    # Check result
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)


def test_preprocess_custom_size():
    """Test preprocessing with custom size."""
    # Create test image
    test_image = Image.new('RGB', (300, 300), (128, 128, 128))
    
    # Preprocess with custom size
    tensor = preprocess(test_image, size=128)
    
    # Check result
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 128, 128)


def test_preprocess_from_path():
    """Test preprocessing from file path."""
    # Create temporary image file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        try:
            # Create and save test image
            test_image = Image.new('RGB', (200, 200), (64, 128, 192))
            test_image.save(tmp.name, 'JPEG')
            
            # Preprocess
            tensor = preprocess(tmp.name)
            
            # Check result
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == (1, 3, 224, 224)
            
        finally:
            # Clean up
            Path(tmp.name).unlink(missing_ok=True)


def test_preprocess_batch():
    """Test batch preprocessing."""
    # Create test images
    test_images = [
        Image.new('RGB', (200, 200), (255, 0, 0)),
        Image.new('RGB', (200, 200), (0, 255, 0)),
        Image.new('RGB', (200, 200), (0, 0, 255))
    ]
    
    # Preprocess batch
    batch_tensor = preprocess_batch(test_images)
    
    # Check result
    assert isinstance(batch_tensor, torch.Tensor)
    assert batch_tensor.shape == (3, 3, 224, 224)  # 3 images, 3 channels, 224x224


def test_preprocess_batch_with_white_balance():
    """Test batch preprocessing with white balance."""
    # Create test images
    test_images = [
        Image.new('RGB', (200, 200), (200, 100, 100)),
        Image.new('RGB', (200, 200), (100, 200, 100)),
        Image.new('RGB', (200, 200), (100, 100, 200))
    ]
    
    # Preprocess batch with white balance
    batch_tensor = preprocess_batch(test_images, white_balance=True)
    
    # Check result
    assert isinstance(batch_tensor, torch.Tensor)
    assert batch_tensor.shape == (3, 3, 224, 224)


def test_preprocess_tensor_input():
    """Test preprocessing with tensor input."""
    # Create test tensor
    test_tensor = torch.randn(3, 300, 300)
    
    # Preprocess
    result_tensor = preprocess(test_tensor)
    
    # Check result
    assert isinstance(result_tensor, torch.Tensor)
    assert result_tensor.shape == (1, 3, 224, 224)


def test_preprocess_tensor_with_batch_dim():
    """Test preprocessing with tensor that already has batch dimension."""
    # Create test tensor with batch dimension
    test_tensor = torch.randn(1, 3, 300, 300)
    
    # Preprocess
    result_tensor = preprocess(test_tensor)
    
    # Check result
    assert isinstance(result_tensor, torch.Tensor)
    assert result_tensor.shape == (1, 3, 224, 224)
