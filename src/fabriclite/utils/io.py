"""I/O utilities for file operations and logging."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.logging import RichHandler


def setup_logger(name: str = "fabriclite", level: int = logging.INFO) -> logging.Logger:
    """Setup logger with rich formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add rich handler
    console = Console()
    handler = RichHandler(console=console, rich_tracebacks=True)
    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="[%X]"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def safe_create_dir(path: Union[str, Path]) -> Path:
    """Safely create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> bool:
    """
    Download file from URL to destination.
    
    Args:
        url: URL to download from
        destination: Destination path
        chunk_size: Chunk size for streaming download
    
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
        
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_huggingface_model(
    repo_id: str,
    filename: str,
    cache_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    Download model from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID
        filename: Model filename
        cache_dir: Cache directory (uses default if None)
    
    Returns:
        Path to downloaded model file, or None if failed
    """
    try:
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "fabriclite"
        
        # Download repository snapshot
        repo_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        model_path = Path(repo_path) / filename
        if model_path.exists():
            return model_path
        else:
            print(f"Model file {filename} not found in repository {repo_id}")
            return None
            
    except Exception as e:
        print(f"Error downloading model from {repo_id}: {e}")
        return None


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load data from JSON file."""
    path = Path(path)
    
    with open(path, 'r') as f:
        return json.load(f)


def get_model_weights_path(
    model_name: str = "mobilenet_v3_small",
    weights_url: Optional[str] = None,
    cache_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    Get path to model weights, downloading if necessary.
    
    Args:
        model_name: Model name
        weights_url: Custom weights URL
        cache_dir: Cache directory
    
    Returns:
        Path to weights file, or None if not available
    """
    # Check environment variable first
    env_weights = os.getenv("FABRICLITE_WEIGHTS")
    if env_weights and Path(env_weights).exists():
        return Path(env_weights)
    
    # Check cache directory
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "fabriclite"
    
    cache_dir = Path(cache_dir)
    weights_path = cache_dir / f"{model_name}_fabrics.pt"
    
    if weights_path.exists():
        return weights_path
    
    # Try to download from Hugging Face Hub
    if weights_url:
        # Custom URL provided
        if download_file(weights_url, weights_path):
            return weights_path
    else:
        # Try Hugging Face Hub (stub repository)
        repo_id = f"fabriclite/{model_name}-fabrics"
        downloaded_path = download_huggingface_model(
            repo_id=repo_id,
            filename=f"{model_name}_fabrics.pt",
            cache_dir=cache_dir
        )
        if downloaded_path:
            return downloaded_path
    
    return None


def get_file_size(path: Union[str, Path]) -> int:
    """Get file size in bytes."""
    return Path(path).stat().st_size


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"
