## src/predict.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from src.model import TextModel,ImageModel
from src.fusion_model import FusionModel
from src.preprocessing import transformer
from src.data_loader import tokenize_data
from PIL import Image
from pathlib import Path


def prediction_text(model, texts, device):
    model.eval()

    inputs = tokenize_data(texts)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits, _ = model(**inputs)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    return probs.cpu().tolist(), preds.cpu().tolist()

def prediction_image(model, image_files, device):
    model.eval()

    images = []
    for img_file in image_files:
        img = Image.open(img_file).convert("RGB")
        transform=transformer()
        images.append(transform(img))

    image_batch = torch.stack(images).to(device)

    with torch.no_grad():
        logits, _ = model(image_batch)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    return probs.cpu().tolist(), preds.cpu().tolist()

def prediction_text_image(fusion_model,text_model,image_model,texts,image_files,device):

    fusion_model.eval()
    text_model.eval()
    image_model.eval()

    input_texts=tokenize_data(texts)
    input_texts={k:v.to(device) for k,v in input_texts.items()}

    images=[]
    for img_file in image_files:
        img=Image.open(img_file).convert("RGB")
        transform=transformer()
        images.append(transform(img))

    image_batch=torch.stack(images).to(device)
    
    # Replicate text features to match image batch size
    num_images = image_batch.shape[0]
    input_texts = {k: v.repeat(num_images, 1) if v.dim() > 1 else v.repeat(num_images) for k, v in input_texts.items()}

    with torch.no_grad():
        _,text_features=text_model(**input_texts)
        _,image_features=image_model(image_batch)

        logits=fusion_model(text_features,image_features)
        probs=torch.softmax(logits,dim=1)
        preds=torch.argmax(probs,dim=1)

        return probs.cpu().tolist(),preds.cpu().tolist()
    