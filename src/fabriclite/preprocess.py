"""Image preprocessing utilities."""

import io
from pathlib import Path
from typing import Union, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def load_image(x: Union[str, Path, bytes, Image.Image, np.ndarray]) -> Image.Image:
    """Load image from various input types and return PIL Image."""
    if isinstance(x, (str, Path)):
        return Image.open(x).convert("RGB")
    elif isinstance(x, bytes):
        return Image.open(io.BytesIO(x)).convert("RGB")
    elif isinstance(x, Image.Image):
        return x.convert("RGB")
    elif isinstance(x, np.ndarray):
        if x.dtype != np.uint8:
            x = (x * 255).astype(np.uint8)
        return Image.fromarray(x)
    else:
        raise ValueError(f"Unsupported image type: {type(x)}")


def apply_gray_world(img: np.ndarray) -> np.ndarray:
    """Apply gray world white balance correction to image."""
    # Convert to float for calculations
    img_float = img.astype(np.float32)
    
    # Calculate mean for each channel
    mean_r = np.mean(img_float[:, :, 0])
    mean_g = np.mean(img_float[:, :, 1])
    mean_b = np.mean(img_float[:, :, 2])
    
    # Calculate gray world average
    gray_world = (mean_r + mean_g + mean_b) / 3.0
    
    # Avoid division by zero
    if mean_r > 0:
        img_float[:, :, 0] *= gray_world / mean_r
    if mean_g > 0:
        img_float[:, :, 1] *= gray_world / mean_g
    if mean_b > 0:
        img_float[:, :, 2] *= gray_world / mean_b
    
    # Clip values and convert back to uint8
    img_balanced = np.clip(img_float, 0, 255).astype(np.uint8)
    
    return img_balanced


def preprocess(
    img: Union[str, Path, bytes, Image.Image, np.ndarray, torch.Tensor],
    white_balance: bool = False,
    size: int = 224
) -> torch.Tensor:
    """
    Preprocess image for model inference.
    
    Args:
        img: Input image in various formats
        white_balance: Whether to apply gray world white balance
        size: Target size for center crop (default 224)
    
    Returns:
        Preprocessed tensor ready for model input
    """
    # Handle tensor input - apply transforms if needed
    if isinstance(img, torch.Tensor):
        if img.dim() == 3:
            img = img.unsqueeze(0)  # Add batch dimension
        
        # If tensor is not the right size, apply transforms
        if img.shape[-2:] != (size, size):
            # Convert tensor to PIL for processing
            # Denormalize if needed (assuming ImageNet normalization)
            if img.min() < 0:  # Likely normalized
                img_denorm = img * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                img_denorm = torch.clamp(img_denorm, 0, 1)
            else:
                img_denorm = img
            
            # Convert to PIL
            img_pil = transforms.ToPILImage()(img_denorm.squeeze(0))
            
            # Apply transforms
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            tensor = transform(img_pil)
            return tensor.unsqueeze(0)
        
        return img
    
    # Load image
    pil_img = load_image(img)
    
    # Apply white balance if requested
    if white_balance:
        img_array = np.array(pil_img)
        img_balanced = apply_gray_world(img_array)
        pil_img = Image.fromarray(img_balanced)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Apply transforms
    tensor = transform(pil_img)
    
    # Add batch dimension
    return tensor.unsqueeze(0)


def preprocess_batch(
    images: list,
    white_balance: bool = False,
    size: int = 224
) -> torch.Tensor:
    """
    Preprocess a batch of images.
    
    Args:
        images: List of images in various formats
        white_balance: Whether to apply gray world white balance
        size: Target size for center crop
    
    Returns:
        Batch tensor ready for model input
    """
    tensors = []
    for img in images:
        tensor = preprocess(img, white_balance=white_balance, size=size)
        tensors.append(tensor.squeeze(0))  # Remove batch dim
    
    return torch.stack(tensors)
