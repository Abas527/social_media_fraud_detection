from fastapi import FastAPI, UploadFile, HTTPException
from fastapi import Form, File
from pydantic import BaseModel
import torch
from src.model import ImageModel,TextModel
from src.fusion_model import FusionModel
from typing import List, Optional
from src.predict import prediction_text,prediction_image,prediction_text_image
from pathlib import Path
import os
import tempfile

class response_model(BaseModel):
    probs: List[List[float]]
    preds: List[int]
    mode: str


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    text_model=TextModel().to(device)
    image_model=ImageModel().to(device)
    fusion_model=FusionModel().to(device)

    text_model.load_state_dict(torch.load("models/text_model.pth",weights_only=True))
    image_model.load_state_dict(torch.load("models/image_model.pth",weights_only=True))
    fusion_model.load_state_dict(torch.load("models/fusion_model.pth",weights_only=True))

    text_model.eval()
    image_model.eval()
    fusion_model.eval()

    return text_model,image_model,fusion_model


def prediction_system(text_model,image_model,fusion_model,device,text_input:Optional[List[str]],images:Optional[List[bytes]]):
    
    if text_input and not images:
        probs,preds=prediction_text(text_model,text_input,device)
        mode="text_only"
    
    elif images and not text_input:
        probs,preds=prediction_image(image_model,images,device)
        mode="image_only"

    elif text_input and images:
        probs,preds=prediction_text_image(fusion_model,text_model,image_model,text_input,images,device)
        mode="text_image"
    else:
        raise ValueError("At least one of the input should be provided.")

    return {
        "probs":probs,
        "preds":preds,
        "mode":mode
    }

app=FastAPI()

text_model,image_model,fusion_model=load_model()

@app.post("/predict/text", response_model=response_model)
async def predict(
    text_input: Optional[List[str]] = Form(None, description="List of text inputs for fraud detection")
):
    """
    Predict social media fraud detection using text only.
    
    - **text_input**: List of text strings to analyze
    - Returns predictions with probabilities and the mode used (text_only)
    """
    
    # Validate that text input is provided
    if not text_input:
        raise HTTPException(
            status_code=400, 
            detail="text_input must be provided."
        )
    
    try:
        # Get predictions from the prediction system
        result = prediction_system(text_model, image_model, fusion_model, device, text_input, None)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/image", response_model=response_model)
async def predict_image(
    images: List[UploadFile] = File(..., description="List of image files for fraud detection")
):
    """
    Predict social media fraud detection using images only.
    
    - **images**: List of image files to analyze
    - Returns predictions with probabilities and the mode used (image_only)
    """
    
    if not images:
        raise HTTPException(
            status_code=400,
            detail="At least one image file must be provided."
        )
    
    image_files = []

    # Process uploaded images
    try:
        for image in images:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp:
                content = await image.read()
                if not content:
                    raise HTTPException(status_code=400, detail="Uploaded file is empty.")
                tmp.write(content)
                tmp_path = tmp.name
                image_files.append(tmp_path)

        # Get predictions from the prediction system
        result = prediction_system(text_model, image_model, fusion_model, device, None, image_files)
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary image files
        for tmp_path in image_files:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.post("/predict/fusion", response_model=response_model)
async def predict_fusion(
    text_input: List[str] = Form(..., description="List of text inputs for fraud detection"),
    images: List[UploadFile] = File(..., description="List of image files for fraud detection")
):
    """
    Predict social media fraud detection using both text and images.
    
    - **text_input**: List of text strings to analyze
    - **images**: List of image files to analyze
    - Returns predictions with probabilities and the mode used (text_image)
    """
    
    if not text_input or not images:
        raise HTTPException(
            status_code=400,
            detail="Both text_input and images must be provided."
        )
    
    image_files = []

    # Process uploaded images
    try:
        for image in images:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp:
                content = await image.read()
                if not content:
                    raise HTTPException(status_code=400, detail="Uploaded file is empty.")
                tmp.write(content)
                tmp_path = tmp.name
                image_files.append(tmp_path)

        # Get predictions from the prediction system
        result = prediction_system(text_model, image_model, fusion_model, device, text_input, image_files)
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary image files
        for tmp_path in image_files:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

@app.get("/")
async def root():
    return {"message":"Welcome to the Social Media Fraud Detection API"}