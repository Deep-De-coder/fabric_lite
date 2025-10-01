"""Tests for CLI infer command structured outputs."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from fabriclite.cli import app
from fabriclite.constants import CLASS_NAMES, CSV_HEADER
from fabriclite.formatting import build_record


@pytest.fixture
def sample_image():
    """Create a temporary sample image for testing."""
    from PIL import Image
    
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color='red')
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img.save(tmp.name, 'JPEG')
        yield Path(tmp.name)
    
    # Cleanup
    Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def mock_classifier():
    """Mock classifier with predictable outputs."""
    with patch('fabriclite.cli.FabricClassifier') as mock:
        instance = mock.from_pretrained.return_value
        
        # Mock predict_proba to return predictable probabilities
        def mock_predict_proba(image, white_balance=False):
            return {
                "cotton": 0.4,
                "denim": 0.3,
                "leather": 0.1,
                "linen": 0.05,
                "silk": 0.05,
                "synthetic": 0.05,
                "velvet": 0.03,
                "wool": 0.02
            }
        
        instance.predict_proba.side_effect = mock_predict_proba
        yield instance


def test_infer_json_output(sample_image, mock_classifier):
    """Test infer command with --json flag."""
    runner = CliRunner()
    result = runner.invoke(app, ["infer", str(sample_image), "--json"])
    
    assert result.exit_code == 0
    
    # Parse JSON output
    output = json.loads(result.stdout)
    
    # Check required keys
    assert "image" in output
    assert "predicted_label" in output
    assert "confidence" in output
    assert "topk" in output
    assert "probs" in output
    
    # Check predicted label is valid
    assert output["predicted_label"] in CLASS_NAMES
    
    # Check confidence is valid
    assert 0.0 <= output["confidence"] <= 1.0
    
    # Check topk structure
    assert isinstance(output["topk"], list)
    assert len(output["topk"]) <= 3  # default topk=3
    for item in output["topk"]:
        assert "label" in item
        assert "prob" in item
        assert item["label"] in CLASS_NAMES
        assert 0.0 <= item["prob"] <= 1.0
    
    # Check probs structure and ordering
    assert list(output["probs"].keys()) == CLASS_NAMES
    for prob in output["probs"].values():
        assert 0.0 <= prob <= 1.0


def test_infer_json_pretty_output(sample_image, mock_classifier):
    """Test infer command with --json --pretty flags."""
    runner = CliRunner()
    result = runner.invoke(app, ["infer", str(sample_image), "--json", "--pretty"])
    
    assert result.exit_code == 0
    
    # Check that output contains newlines and spaces (pretty formatting)
    assert '\n' in result.stdout
    assert '  ' in result.stdout  # indentation
    
    # Parse JSON output
    output = json.loads(result.stdout)
    assert "image" in output
    assert "predicted_label" in output


def test_infer_csv_output(sample_image, mock_classifier):
    """Test infer command with --csv flag."""
    runner = CliRunner()
    result = runner.invoke(app, ["infer", str(sample_image), "--csv"])
    
    assert result.exit_code == 0
    
    lines = result.stdout.strip().split('\n')
    assert len(lines) == 2  # header + data row
    
    # Check header (strip any Windows line endings)
    header = lines[0].strip().split(',')
    assert header == CSV_HEADER
    
    # Check data row (strip any Windows line endings)
    data_row = lines[1].strip().split(',')
    assert len(data_row) == len(CSV_HEADER)
    
    # Check predicted label is valid
    predicted_label_idx = CSV_HEADER.index("predicted_label")
    assert data_row[predicted_label_idx] in CLASS_NAMES
    
    # Check confidence is valid
    confidence_idx = CSV_HEADER.index("confidence")
    confidence = float(data_row[confidence_idx])
    assert 0.0 <= confidence <= 1.0
    
    # Check probabilities sum to approximately 1.0
    prob_indices = [CSV_HEADER.index(cls) for cls in CLASS_NAMES]
    probs = [float(data_row[i]) for i in prob_indices]
    assert abs(sum(probs) - 1.0) < 1e-3


def test_infer_mutually_exclusive_flags(sample_image, mock_classifier):
    """Test that --json and --csv flags are mutually exclusive."""
    runner = CliRunner()
    result = runner.invoke(app, ["infer", str(sample_image), "--json", "--csv"])
    
    assert result.exit_code != 0
    assert "Use either --json or --csv, not both" in result.stdout


def test_infer_json_with_output_file(sample_image, mock_classifier):
    """Test infer command with --json and --output file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        runner = CliRunner()
        result = runner.invoke(app, ["infer", str(sample_image), "--json", "--output", str(tmp_path)])
        
        assert result.exit_code == 0
        
        # Check file was created and contains valid JSON
        assert tmp_path.exists()
        with open(tmp_path) as f:
            output = json.load(f)
        
        assert "image" in output
        assert "predicted_label" in output
        assert "confidence" in output
        assert "topk" in output
        assert "probs" in output
    
    finally:
        tmp_path.unlink(missing_ok=True)


def test_infer_csv_with_output_file(sample_image, mock_classifier):
    """Test infer command with --csv and --output file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        runner = CliRunner()
        result = runner.invoke(app, ["infer", str(sample_image), "--csv", "--output", str(tmp_path)])
        
        assert result.exit_code == 0
        
        # Check file was created and contains valid CSV
        assert tmp_path.exists()
        with open(tmp_path) as f:
            lines = f.readlines()
        
        assert len(lines) == 2  # header + data row
        assert lines[0].strip().split(',') == CSV_HEADER
    
    finally:
        tmp_path.unlink(missing_ok=True)


def test_infer_default_behavior(sample_image, mock_classifier):
    """Test that default behavior (no --json/--csv) still works."""
    runner = CliRunner()
    result = runner.invoke(app, ["infer", str(sample_image)])
    
    assert result.exit_code == 0
    
    # Should contain Rich table output (not JSON/CSV)
    assert "Fabric Classification Results" in result.stdout
    assert "Fabric Type" in result.stdout
    assert "Probability" in result.stdout


def test_infer_with_topk(sample_image, mock_classifier):
    """Test infer command with custom --topk value."""
    runner = CliRunner()
    result = runner.invoke(app, ["infer", str(sample_image), "--json", "--topk", "5"])
    
    assert result.exit_code == 0
    
    output = json.loads(result.stdout)
    assert len(output["topk"]) == 5
    
    # Check topk is sorted by probability (descending)
    probs = [item["prob"] for item in output["topk"]]
    assert probs == sorted(probs, reverse=True)


def test_infer_with_white_balance(sample_image, mock_classifier):
    """Test infer command with --wb flag."""
    runner = CliRunner()
    result = runner.invoke(app, ["infer", str(sample_image), "--json", "--wb"])
    
    assert result.exit_code == 0
    
    # Verify that predict_proba was called with white_balance=True
    mock_classifier.predict_proba.assert_called_with(sample_image, white_balance=True)
