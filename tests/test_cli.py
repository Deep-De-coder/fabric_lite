"""Tests for FabricLite CLI."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from fabriclite.cli import app

runner = CliRunner()


def test_cli_help():
    """Test CLI help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "FabricLite: A tiny fabric/material classifier for garments" in result.output


def test_infer_command_help():
    """Test infer command help."""
    result = runner.invoke(app, ["infer", "--help"])
    assert result.exit_code == 0
    assert "Infer fabric type from a single image" in result.output


def test_batch_command_help():
    """Test batch command help."""
    result = runner.invoke(app, ["batch", "--help"])
    assert result.exit_code == 0
    assert "Batch inference on folder of images" in result.output


def test_train_command_help():
    """Test train command help."""
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "Train fabric classifier" in result.output


def test_eval_command_help():
    """Test eval command help."""
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "Evaluate model on test data" in result.output


def test_calibrate_command_help():
    """Test calibrate command help."""
    result = runner.invoke(app, ["calibrate", "--help"])
    assert result.exit_code == 0
    assert "Calibrate model using temperature scaling" in result.output


def test_export_command_help():
    """Test export command help."""
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0
    assert "Export model to ONNX or TorchScript format" in result.output


def test_infer_with_nonexistent_file():
    """Test infer command with nonexistent file."""
    result = runner.invoke(app, ["infer", "nonexistent.jpg"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_batch_with_nonexistent_folder():
    """Test batch command with nonexistent folder."""
    result = runner.invoke(app, ["batch", "nonexistent_folder"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_train_with_nonexistent_dirs():
    """Test train command with nonexistent directories."""
    result = runner.invoke(app, ["train", "nonexistent_train", "nonexistent_val"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_eval_with_nonexistent_files():
    """Test eval command with nonexistent files."""
    result = runner.invoke(app, ["eval", "nonexistent_data", "nonexistent_weights.pt"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_calibrate_with_nonexistent_files():
    """Test calibrate command with nonexistent files."""
    result = runner.invoke(app, ["calibrate", "nonexistent_data", "nonexistent_weights.pt"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_export_with_nonexistent_weights():
    """Test export command with nonexistent weights."""
    result = runner.invoke(app, ["export", "nonexistent_weights.pt"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


@patch('fabriclite.cli.FabricClassifier')
def test_infer_with_mock_classifier(mock_classifier_class):
    """Test infer command with mocked classifier."""
    # Create temporary image file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        try:
            # Create a simple test image
            from PIL import Image
            test_image = Image.new('RGB', (224, 224), (128, 128, 128))
            test_image.save(tmp.name, 'JPEG')
            
            # Mock classifier
            mock_classifier = mock_classifier_class.from_pretrained.return_value
            mock_classifier.predict.return_value = {
                'cotton': 0.8,
                'denim': 0.15,
                'silk': 0.05
            }
            
            # Run infer command
            result = runner.invoke(app, ["infer", tmp.name])
            
            # Check that command succeeded
            assert result.exit_code == 0
            assert "cotton" in result.output
            assert "0.800" in result.output
            
        finally:
            # Clean up
            Path(tmp.name).unlink(missing_ok=True)


def test_infer_with_invalid_image():
    """Test infer command with invalid image file."""
    # Create temporary file with invalid content
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        try:
            # Write invalid content
            tmp.write(b"not an image")
            tmp.flush()
            
            # Run infer command
            result = runner.invoke(app, ["infer", tmp.name])
            
            # Should fail gracefully
            assert result.exit_code == 1
            
        finally:
            # Clean up
            Path(tmp.name).unlink(missing_ok=True)


def test_export_invalid_format():
    """Test export command with invalid format."""
    # Create temporary weights file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        try:
            # Write dummy content
            tmp.write(b"dummy weights")
            tmp.flush()
            
            # Run export command with invalid format
            result = runner.invoke(app, ["export", tmp.name, "--format", "invalid"])
            
            # Should fail
            assert result.exit_code == 1
            assert "Unsupported format" in result.output
            
        finally:
            # Clean up
            Path(tmp.name).unlink(missing_ok=True)
