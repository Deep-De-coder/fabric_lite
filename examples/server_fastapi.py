"""FastAPI microservice demo for FabricLite."""

import io
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from fabriclite import FabricClassifier, FABRIC_LABELS

# Initialize FastAPI app
app = FastAPI(
    title="FabricLite API",
    description="A tiny fabric/material classifier for garments",
    version="0.1.0"
)

# Global classifier instance (lazy loaded)
classifier: Optional[FabricClassifier] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    labels: List[str]


class PredictionResponse(BaseModel):
    """Prediction response model."""
    label: str
    probs: Dict[str, float]
    temperature: float


def get_classifier() -> FabricClassifier:
    """Get or create classifier instance."""
    global classifier
    
    if classifier is None:
        # Check for custom weights path
        weights_path = os.getenv("FABRICLITE_WEIGHTS")
        
        if weights_path and Path(weights_path).exists():
            # Load custom weights
            classifier = FabricClassifier.from_pretrained()
            classifier.model.load_state_dict(
                torch.load(weights_path, map_location="cpu")
            )
        else:
            # Use default pretrained model
            classifier = FabricClassifier.from_pretrained()
    
    return classifier


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        labels=FABRIC_LABELS
    )


@app.post("/infer", response_model=PredictionResponse)
async def infer_fabric(
    file: UploadFile = File(...),
    topk: int = 3,
    white_balance: bool = False
):
    """
    Infer fabric type from uploaded image.
    
    Args:
        file: Uploaded image file
        topk: Number of top predictions to return
        white_balance: Whether to apply white balance correction
    
    Returns:
        Prediction results with probabilities
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read file content
        content = await file.read()
        
        # Get classifier
        model = get_classifier()
        
        # Make prediction
        result = model.predict(
            content,
            topk=topk,
            white_balance=white_balance
        )
        
        # Get top prediction
        top_label = max(result, key=result.get)
        
        return PredictionResponse(
            label=top_label,
            probs=result,
            temperature=model.temperature
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/infer_batch")
async def infer_batch(
    files: List[UploadFile] = File(...),
    white_balance: bool = False
):
    """
    Batch inference on multiple images.
    
    Args:
        files: List of uploaded image files
        white_balance: Whether to apply white balance correction
    
    Returns:
        List of prediction results
    """
    try:
        # Validate files
        for file in files:
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} must be an image"
                )
        
        # Get classifier
        model = get_classifier()
        
        # Process images
        results = []
        for file in files:
            try:
                content = await file.read()
                result = model.predict(content, white_balance=white_balance)
                
                top_label = max(result, key=result.get)
                
                results.append({
                    "filename": file.filename,
                    "label": top_label,
                    "probs": result
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/labels")
async def get_labels():
    """Get available fabric labels."""
    return {"labels": FABRIC_LABELS}


@app.get("/model_info")
async def get_model_info():
    """Get model information."""
    model = get_classifier()
    
    return {
        "model_type": "MobileNetV3-Small",
        "num_classes": len(FABRIC_LABELS),
        "labels": FABRIC_LABELS,
        "temperature": model.temperature,
        "device": model.device
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "server_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
