"""Model export utilities for ONNX and TorchScript."""

import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.onnx
from torch.jit import ScriptModule


def to_onnx(
    model: torch.nn.Module,
    out_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    opset: int = 17,
    dynamic_axes: Optional[dict] = None
) -> None:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        out_path: Output path for ONNX file
        input_shape: Input tensor shape (batch_size, channels, height, width)
        opset: ONNX opset version
        dynamic_axes: Dynamic axes configuration
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Default dynamic axes for batch dimension
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print(f"Model exported to ONNX: {out_path}")


def to_torchscript(
    model: torch.nn.Module,
    out_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    method: str = "trace"
) -> ScriptModule:
    """
    Export PyTorch model to TorchScript format.
    
    Args:
        model: PyTorch model to export
        out_path: Output path for TorchScript file
        input_shape: Input tensor shape for tracing
        method: Export method ("trace" or "script")
    
    Returns:
        TorchScript model
    """
    model.eval()
    
    if method == "trace":
        # Create dummy input for tracing
        dummy_input = torch.randn(input_shape)
        
        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
        
        # Save traced model
        traced_model.save(out_path)
        print(f"Model exported to TorchScript (traced): {out_path}")
        
        return traced_model
    
    elif method == "script":
        # Script the model
        scripted_model = torch.jit.script(model)
        
        # Save scripted model
        scripted_model.save(out_path)
        print(f"Model exported to TorchScript (scripted): {out_path}")
        
        return scripted_model
    
    else:
        raise ValueError(f"Unknown export method: {method}")


def to_tflite(
    model: torch.nn.Module,
    out_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)
) -> None:
    """
    Export PyTorch model to TensorFlow Lite format (stub implementation).
    
    Note: This is a stub implementation. For production use, you would need
    to convert through TensorFlow using tools like tf2onnx or similar.
    
    Args:
        model: PyTorch model to export
        out_path: Output path for TFLite file
        input_shape: Input tensor shape
    """
    warnings.warn(
        "TFLite export is not fully implemented. This is a stub. "
        "For production use, convert through TensorFlow using tf2onnx or similar tools.",
        UserWarning
    )
    
    # Stub implementation - just create a placeholder file
    with open(out_path, 'w') as f:
        f.write("# TFLite export stub - not implemented\n")
        f.write(f"# Model input shape: {input_shape}\n")
        f.write("# Use tf2onnx or similar tools for actual TFLite conversion\n")
    
    print(f"TFLite export stub created: {out_path}")


def verify_onnx_model(onnx_path: str) -> bool:
    """
    Verify ONNX model can be loaded and run.
    
    Args:
        onnx_path: Path to ONNX model file
    
    Returns:
        True if model is valid, False otherwise
    """
    try:
        import onnx
        import onnxruntime as ort
        
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Test inference with ONNX Runtime
        session = ort.InferenceSession(onnx_path)
        
        # Get input details
        input_details = session.get_inputs()[0]
        input_shape = input_details.shape
        input_name = input_details.name
        
        # Create dummy input
        dummy_input = {input_name: torch.randn(input_shape).numpy()}
        
        # Run inference
        outputs = session.run(None, dummy_input)
        
        print(f"ONNX model verification successful: {onnx_path}")
        return True
        
    except ImportError:
        print("ONNX or ONNX Runtime not available for verification")
        return False
    except Exception as e:
        print(f"ONNX model verification failed: {e}")
        return False


def verify_torchscript_model(ts_path: str) -> bool:
    """
    Verify TorchScript model can be loaded and run.
    
    Args:
        ts_path: Path to TorchScript model file
    
    Returns:
        True if model is valid, False otherwise
    """
    try:
        # Load TorchScript model
        model = torch.jit.load(ts_path)
        
        # Test inference
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"TorchScript model verification successful: {ts_path}")
        return True
        
    except Exception as e:
        print(f"TorchScript model verification failed: {e}")
        return False
