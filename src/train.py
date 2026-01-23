## src/train.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import TextModel
from src.data_loader import create_dataloaders
import torch.nn as nn
import pandas as pd
from torch.optim import AdamW
from tqdm import tqdm
import logging
import mlflow
from src.mlflow_utils import setup_experiment, log_metrics, log_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


print(torch.cuda.is_available())


criterion=nn.CrossEntropyLoss()
model=TextModel()
optimizer=AdamW(model.parameters(),lr=2e-5)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
df=pd.read_csv("data/processed/text/text_data.csv")
train_loader,val_loader,test_loader=create_dataloaders(df)




def train_model(model,optimizer,criterion,train_loader,device,epochs=3):
    logger.info("Starting text model training...")
    
    for epoch in range(epochs):
            model.train()
            train_los=0
            for batch in train_loader:
                optimizer.zero_grad()

                input_ids=batch["input_ids"].to(device)
                attention_mask=batch["attention_mask"].to(device)
                labels=batch["labels"].to(device)

                outputs,_=model(input_ids=input_ids,attention_mask=attention_mask)
                loss=criterion(outputs,labels)

                loss.backward()
                optimizer.step()
                train_los+=loss.item()
            
            avg_train_loss=train_los/len(train_loader)
            mlflow.log_metric("text_train_loss", avg_train_loss, step=epoch)
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}")
    
    torch.save(model.state_dict(),"models/text_model.pth")
    mlflow.log_artifact("models/text_model.pth")
    
    


# def tran_model for images
from src.model import ImageModel
from src.data_loader import create_image_loader
from pathlib import Path

data_dir=Path("data/raw/image")
image_train_loader,image_val_loader=create_image_loader(data_dir)
image_model=ImageModel()
optimizer=AdamW(image_model.parameters(),lr=2e-5)
image_model.to(device)


def train_image_model(model,optimizer,criterion,train_loader,device,epochs=5):
    logger.info("Starting image model training...")
    
    for epoch in range(epochs):
            model.train()
            train_loss=0
            correct=0
            total=0

            for image,labels in tqdm(train_loader):
                optimizer.zero_grad()

                image=image.to(device)
                labels=labels.to(device)

                output,_=model(image)
                loss=criterion(output,labels)

                loss.backward()
                optimizer.step()

                train_loss+=loss.item()
                _,predicted=torch.max(output.data,1)
                correct+= (predicted==labels).sum().item()
                total+=labels.size(0)
            
            avg_train_loss=train_loss/len(train_loader)
            accuracy = 100*correct/total
            mlflow.log_metric("image_train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("image_train_accuracy", accuracy, step=epoch)
            
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}, Training Accuracy: {accuracy}%")
    
    logger.info("Image model training completed")
    torch.save(model.state_dict(),"models/image_model.pth")
    mlflow.log_artifact("models/image_model.pth")
    
    


def main():
    setup_experiment("Social Media Fraud Detection - Text and Image Models")
    
    with mlflow.start_run():
        log_params({
            "text_epochs": 3,
            "image_epochs": 5,
            "learning_rate": 2e-5,
            "device": str(device)
        })
        
        train_model(model,optimizer,criterion,train_loader,device,epochs=3)
        train_image_model(image_model,optimizer,criterion,image_train_loader,device,epochs=5)

if __name__=="__main__":
    main()

