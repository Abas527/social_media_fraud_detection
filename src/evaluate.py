## src/evaluate.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.model import TextModel
from src.data_loader import create_dataloaders
import torch.nn as nn
import pandas as pd
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,recall_score, precision_score
from torchvision import models
import tqdm
import mlflow
from src.mlflow_utils import setup_experiment
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

setup_experiment("Text_and_Image_Classification_evaluation")


criterion=nn.CrossEntropyLoss()
model=TextModel()
optimizer=AdamW(model.parameters(),lr=2e-5)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load("models/text_model.pth", map_location=device, weights_only=True))
df=pd.read_csv("data/processed/text/text_data.csv")
train_loader,val_loader,test_loader=create_dataloaders(df)


def evaluate_model(model,criterion,val_loader,device):
    model.eval()

    val_loss=0
    correct=0
    total=0

    with torch.no_grad():
        label_list=[]
        pred_list=[]
        for batch, labels in val_loader:
            input_ids=batch["input_ids"].to(device)
            attention_mask=batch["attention_mask"].to(device)
            labels=labels.to(device)

            outputs,_=model(input_ids=input_ids,attention_mask=attention_mask)
            loss=criterion(outputs,labels)

            val_loss+=loss.item()
            _,predicted=torch.max(outputs,1)
            correct+=(predicted==labels).sum().item()
            total+=labels.size(0)
            label_list.extend(labels.cpu().numpy())
            pred_list.extend(predicted.cpu().numpy())
        
        classification_rep=classification_report(label_list,pred_list)
        
        
        precision = precision_score(label_list, pred_list, average='weighted', zero_division=0)
        recall = recall_score(label_list, pred_list, average='weighted', zero_division=0)
           

    avg_val_loss=val_loss/len(val_loader)
    accuracy=correct/total

    return avg_val_loss,accuracy,classification_rep,precision,recall


#image model parameters
from model import ImageModel
from data_loader import create_image_loader
from pathlib import Path

data_dir=Path("data/raw/image")
image_train_loader,image_val_loader=create_image_loader(data_dir)
image_model=ImageModel()
image_model.to(device)
image_model.load_state_dict(torch.load("models/image_model.pth", map_location=device, weights_only=True))
optimizer=AdamW(image_model.parameters(),lr=2e-5)




def evaluate_image_model(model,criterion,val_loader,device):
    model.eval()
    val_loss=0
    correct=0
    total=0

    with torch.no_grad():
        label_list=[]
        pred_list=[]

        for image,labels in val_loader:
            image=image.to(device)
            labels=labels.to(device)

            outputs,_=model(image)
            loss=criterion(outputs,labels)

            val_loss+=loss.item()
            _,predicted=torch.max(outputs,1)
            correct+=(predicted==labels).sum().item()
            total+=labels.size(0)
            label_list.extend(labels.cpu().numpy())
            pred_list.extend(predicted.cpu().numpy())
        classification_rep=classification_report(label_list,pred_list)
        
        
        precision = precision_score(label_list, pred_list, average='weighted', zero_division=0)
        recall = recall_score(label_list, pred_list, average='weighted', zero_division=0)
    avg_val_loss=val_loss/len(val_loader)
    accuracy=correct/total
    return avg_val_loss,accuracy,classification_rep,precision,recall

def main():
    # Evaluate Text Model
    with mlflow.start_run(run_name="text_model_evaluation"):
        logger.info("Starting text model evaluation...")
        val_loss,accuracy,classification_rep,precision,recall=evaluate_model(model,criterion,val_loader,device)
        print(f"Validation Loss: {val_loss}, Accuracy: {accuracy}")
        print("Classification Report:")
        print(classification_rep)
        
        # Log metrics
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Log classification report as artifact
        mlflow.log_text(classification_rep, "text_classification_report.txt")
        logger.info("Text model evaluation completed and logged")

    # Evaluate Image Model
    with mlflow.start_run(run_name="image_model_evaluation"):
        logger.info("Starting image model evaluation...")
        img_val_loss,img_accuracy,img_classification_rep,img_precision,img_recall=evaluate_image_model(image_model,criterion,image_val_loader,device)
        print(f"Image Validation Loss: {img_val_loss}, Image Accuracy: {img_accuracy}")
        print("Image Classification Report:")
        print(img_classification_rep)
        
        # Log metrics
        mlflow.log_metric("val_loss", img_val_loss)
        mlflow.log_metric("accuracy", img_accuracy)
        mlflow.log_metric("precision", img_precision)
        mlflow.log_metric("recall", img_recall)
        
        # Log classification report as artifact
        mlflow.log_text(img_classification_rep, "image_classification_report.txt")
        logger.info("Image model evaluation completed and logged")

if __name__=="__main__":
    main()