"""Tests for CLI batch command structured outputs."""

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from fabriclite.cli import app
from fabriclite.constants import CLASS_NAMES, CSV_HEADER


@pytest.fixture
def sample_images():
    """Create temporary sample images for testing."""
    from PIL import Image
    
    images = []
    temp_files = []
    
    # Create 3 test images
    for i in range(3):
        img = Image.new('RGB', (224, 224), color=('red', 'green', 'blue')[i])
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img.save(tmp.name, 'JPEG')
            images.append(Path(tmp.name))
            temp_files.append(tmp.name)
    
    yield images
    
    # Cleanup
    for tmp_file in temp_files:
        Path(tmp_file).unlink(missing_ok=True)


@pytest.fixture
def sample_folder(sample_images):
    """Create a temporary folder with sample images."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        folder = Path(tmp_dir)
        
        # Copy images to folder
        for i, img_path in enumerate(sample_images):
            new_path = folder / f"image_{i}.jpg"
            new_path.write_bytes(img_path.read_bytes())
        
        yield folder


@pytest.fixture
def mock_classifier():
    """Mock classifier with predictable outputs."""
    with patch('fabriclite.cli.FabricClassifier') as mock:
        instance = mock.from_pretrained.return_value
        
        # Mock predict_proba to return predictable probabilities
        def mock_predict_proba(image, white_balance=False):
            # Return different probabilities based on image name for testing
            if "image_0" in str(image):
                return {
                    "cotton": 0.5, "denim": 0.2, "leather": 0.1, "linen": 0.05,
                    "silk": 0.05, "synthetic": 0.05, "velvet": 0.03, "wool": 0.02
                }
            elif "image_1" in str(image):
                return {
                    "cotton": 0.2, "denim": 0.4, "leather": 0.15, "linen": 0.1,
                    "silk": 0.05, "synthetic": 0.05, "velvet": 0.03, "wool": 0.02
                }
            else:  # image_2
                return {
                    "cotton": 0.1, "denim": 0.1, "leather": 0.3, "linen": 0.2,
                    "silk": 0.1, "synthetic": 0.1, "velvet": 0.05, "wool": 0.05
                }
        
        instance.predict_proba.side_effect = mock_predict_proba
        yield instance


def test_batch_jsonl_output(sample_folder, mock_classifier):
    """Test batch command with --json and .jsonl output."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        runner = CliRunner()
        result = runner.invoke(app, ["batch", str(sample_folder), "--json", "--output", str(tmp_path)])
        
        assert result.exit_code == 0
        
        # Check file was created
        assert tmp_path.exists()
        
        # Read JSONL file
        records = []
        with open(tmp_path) as f:
            for line in f:
                records.append(json.loads(line.strip()))
        
        # Should have 3 records (one per image)
        assert len(records) == 3
        
        # Check each record structure
        for record in records:
            assert "image" in record
            assert "predicted_label" in record
            assert "confidence" in record
            assert "topk" in record
            assert "probs" in record
            
            # Check predicted label is valid
            assert record["predicted_label"] in CLASS_NAMES
            
            # Check confidence is valid
            assert 0.0 <= record["confidence"] <= 1.0
            
            # Check probs structure and ordering
            assert list(record["probs"].keys()) == CLASS_NAMES
            for prob in record["probs"].values():
                assert 0.0 <= prob <= 1.0
    
    finally:
        tmp_path.unlink(missing_ok=True)


def test_batch_json_array_output(sample_folder, mock_classifier):
    """Test batch command with --json and .json output (array format)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        runner = CliRunner()
        result = runner.invoke(app, ["batch", str(sample_folder), "--json", "--output", str(tmp_path)])
        
        assert result.exit_code == 0
        
        # Check file was created
        assert tmp_path.exists()
        
        # Read JSON array
        with open(tmp_path) as f:
            records = json.load(f)
        
        # Should be a list with 3 records
        assert isinstance(records, list)
        assert len(records) == 3
        
        # Check each record structure
        for record in records:
            assert "image" in record
            assert "predicted_label" in record
            assert "confidence" in record
            assert "topk" in record
            assert "probs" in record
    
    finally:
        tmp_path.unlink(missing_ok=True)


def test_batch_json_pretty_output(sample_folder, mock_classifier):
    """Test batch command with --json --pretty and .json output."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        runner = CliRunner()
        result = runner.invoke(app, ["batch", str(sample_folder), "--json", "--pretty", "--output", str(tmp_path)])
        
        assert result.exit_code == 0
        
        # Check file was created
        assert tmp_path.exists()
        
        # Read file and check for pretty formatting
        with open(tmp_path) as f:
            content = f.read()
        
        # Should contain indentation (pretty formatting)
        assert '  ' in content
        assert '\n' in content
        
        # Should still be valid JSON
        records = json.loads(content)
        assert isinstance(records, list)
        assert len(records) == 3
    
    finally:
        tmp_path.unlink(missing_ok=True)


def test_batch_csv_output(sample_folder, mock_classifier):
    """Test batch command with --csv output."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        runner = CliRunner()
        result = runner.invoke(app, ["batch", str(sample_folder), "--csv", "--output", str(tmp_path)])
        
        assert result.exit_code == 0
        
        # Check file was created
        assert tmp_path.exists()
        
        # Read CSV file
        with open(tmp_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Should have header + 3 data rows
        assert len(rows) == 4  # header + 3 images
        
        # Check header
        assert rows[0] == CSV_HEADER
        
        # Check data rows
        for i, row in enumerate(rows[1:], 1):
            assert len(row) == len(CSV_HEADER)
            
            # Check predicted label is valid
            predicted_label_idx = CSV_HEADER.index("predicted_label")
            assert row[predicted_label_idx] in CLASS_NAMES
            
            # Check confidence is valid
            confidence_idx = CSV_HEADER.index("confidence")
            confidence = float(row[confidence_idx])
            assert 0.0 <= confidence <= 1.0
            
            # Check probabilities sum to approximately 1.0
            prob_indices = [CSV_HEADER.index(cls) for cls in CLASS_NAMES]
            probs = [float(row[j]) for j in prob_indices]
            assert abs(sum(probs) - 1.0) < 1e-3
    
    finally:
        tmp_path.unlink(missing_ok=True)


def test_batch_jsonl_stdout(sample_folder, mock_classifier):
    """Test batch command with --json output to stdout (JSONL format)."""
    runner = CliRunner()
    result = runner.invoke(app, ["batch", str(sample_folder), "--json"])
    
    assert result.exit_code == 0
    
    # Parse JSONL from stdout
    lines = result.stdout.strip().split('\n')
    assert len(lines) == 3  # one line per image
    
    records = []
    for line in lines:
        records.append(json.loads(line))
    
    # Check each record structure
    for record in records:
        assert "image" in record
        assert "predicted_label" in record
        assert "confidence" in record
        assert "topk" in record
        assert "probs" in record


def test_batch_csv_stdout(sample_folder, mock_classifier):
    """Test batch command with --csv output to stdout."""
    runner = CliRunner()
    result = runner.invoke(app, ["batch", str(sample_folder), "--csv"])
    
    assert result.exit_code == 0
    
    # Parse CSV from stdout
    lines = result.stdout.strip().split('\n')
    assert len(lines) == 4  # header + 3 data rows
    
    # Check header (strip any Windows line endings)
    assert lines[0].strip().split(',') == CSV_HEADER
    
    # Check data rows
    for line in lines[1:]:
        row = line.strip().split(',')
        assert len(row) == len(CSV_HEADER)
        
        # Check predicted label is valid
        predicted_label_idx = CSV_HEADER.index("predicted_label")
        assert row[predicted_label_idx] in CLASS_NAMES


def test_batch_mutually_exclusive_flags(sample_folder, mock_classifier):
    """Test that --json and --csv flags are mutually exclusive."""
    runner = CliRunner()
    result = runner.invoke(app, ["batch", str(sample_folder), "--json", "--csv"])
    
    assert result.exit_code != 0
    assert "Use either --json or --csv, not both" in result.stdout


def test_batch_default_behavior(sample_folder, mock_classifier):
    """Test that default behavior (no --json/--csv) still works."""
    runner = CliRunner()
    result = runner.invoke(app, ["batch", str(sample_folder)])
    
    assert result.exit_code == 0
    
    # Should contain Rich output (not JSON/CSV)
    assert "Processed" in result.stdout
    assert "images" in result.stdout


def test_batch_with_topk(sample_folder, mock_classifier):
    """Test batch command with custom --topk value."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        runner = CliRunner()
        result = runner.invoke(app, ["batch", str(sample_folder), "--json", "--topk", "5", "--output", str(tmp_path)])
        
        assert result.exit_code == 0
        
        # Read JSONL file
        records = []
        with open(tmp_path) as f:
            for line in f:
                records.append(json.loads(line.strip()))
        
        # Check each record has 5 topk items
        for record in records:
            assert len(record["topk"]) == 5
            
            # Check topk is sorted by probability (descending)
            probs = [item["prob"] for item in record["topk"]]
            assert probs == sorted(probs, reverse=True)
    
    finally:
        tmp_path.unlink(missing_ok=True)


def test_batch_with_white_balance(sample_folder, mock_classifier):
    """Test batch command with --wb flag."""
    runner = CliRunner()
    result = runner.invoke(app, ["batch", str(sample_folder), "--json", "--wb"])
    
    assert result.exit_code == 0
    
    # Verify that predict_proba was called with white_balance=True for each image
    assert mock_classifier.predict_proba.call_count == 3
    for call in mock_classifier.predict_proba.call_args_list:
        assert call[1]['white_balance'] is True


def test_batch_error_handling(sample_folder, mock_classifier):
    """Test batch command error handling for unreadable images."""
    # Mock predict_proba to raise an exception for one image
    def mock_predict_proba_with_error(image, white_balance=False):
        if "image_1" in str(image):
            raise Exception("Cannot read image")
        return {
            "cotton": 0.5, "denim": 0.2, "leather": 0.1, "linen": 0.05,
            "silk": 0.05, "synthetic": 0.05, "velvet": 0.03, "wool": 0.02
        }
    
    mock_classifier.predict_proba.side_effect = mock_predict_proba_with_error
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        runner = CliRunner()
        result = runner.invoke(app, ["batch", str(sample_folder), "--json", "--output", str(tmp_path)])
        
        if result.exit_code != 0:
            print(f"Error output: {result.stdout}")
            print(f"Exception: {result.exception}")
        
        assert result.exit_code == 0
        
        # Read JSONL file
        records = []
        with open(tmp_path) as f:
            for line in f:
                records.append(json.loads(line.strip()))
        
        # Should have 3 records (including error record)
        assert len(records) == 3
        
        # Find the error record
        error_record = None
        for record in records:
            if "error" in record:
                error_record = record
                break
        
        assert error_record is not None
        assert error_record["predicted_label"] == ""
        assert error_record["confidence"] == 0.0
        assert "error" in error_record
        assert "Cannot read image" in error_record["error"]
    
    finally:
        tmp_path.unlink(missing_ok=True)
