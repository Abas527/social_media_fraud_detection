## src/model.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#text model

import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import split_data
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertModel
from torch.utils.data import DataLoader, Dataset
from src.data_loader import create_dataloaders
from pathlib import Path

class TextModel(torch.nn.Module):

    def __init__(self):

        super(TextModel, self).__init__()

        encoder=DistilBertModel.from_pretrained('distilbert-base-uncased')

        for param in encoder.parameters():
            param.requires_grad = False   # Freeze the encoder last layers
        
        self.encoder=encoder
        self.classifier=torch.nn.Linear(self.encoder.config.hidden_size,2) #binary classification
        self.dropout=torch.nn.Dropout(0.1)                                 # dropout layer
    
    def forward(self,input_ids,attention_mask):
        
        outputs=self.encoder(input_ids=input_ids,attention_mask=attention_mask)
        cls_embedding=outputs.last_hidden_state[:,0,:]                         # CLS token embedding
        x=self.dropout(cls_embedding)                                          # dropout
        logits=self.classifier(x)                                            # classification layer
        return logits,cls_embedding                                          # return logits and CLS/Text embedding



# defining the image mode as Resnet 50
from torchvision import models
from torch import nn

class ImageModel(torch.nn.Module):
    def __init__(self):
        super(ImageModel,self).__init__()

        self.model=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for params in self.model.parameters():
            params.requires_grad=False
        
        in_features=self.model.fc.in_features
        
        self.model.fc=torch.nn.Identity()  #removing the last layer

        self.classifier=nn.Sequential(
            nn.Linear(in_features,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,2)
        )
    
    
    
    def forward(self,x):
        features=self.model(x)
        logits=self.classifier(features)
        return logits,features


