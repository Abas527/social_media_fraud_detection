## src/evaluate.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import TextModel
from src.data_loader import create_dataloaders
import torch.nn as nn
import pandas as pd
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,recall_score, precision_score
from torchvision import models
import tqdm
import logging
import mlflow
from src.mlflow_utils import setup_experiment, log_metrics, log_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        classification_rep_dict=classification_report(label_list,pred_list,output_dict=True)
        
        precision = precision_score(label_list, pred_list, average='weighted', zero_division=0)
        recall = recall_score(label_list, pred_list, average='weighted', zero_division=0)
        
        # Log metrics to MLflow
        mlflow.log_metric("text_val_loss", val_loss/len(val_loader))
        mlflow.log_metric("text_val_accuracy", correct/total)
        mlflow.log_metric("text_val_precision", precision)
        mlflow.log_metric("text_val_recall", recall)
        mlflow.log_metric("text_val_f1", classification_rep_dict['weighted avg']['f1-score'])

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
        classification_rep_dict=classification_report(label_list,pred_list,output_dict=True)
        
        precision = precision_score(label_list, pred_list, average='weighted', zero_division=0)
        recall = recall_score(label_list, pred_list, average='weighted', zero_division=0)
        
        # Log metrics to MLflow
        mlflow.log_metric("image_val_loss", val_loss/len(val_loader))
        mlflow.log_metric("image_val_accuracy", correct/total)
        mlflow.log_metric("image_val_precision", precision)
        mlflow.log_metric("image_val_recall", recall)
        mlflow.log_metric("image_val_f1", classification_rep_dict['weighted avg']['f1-score'])
    
    avg_val_loss=val_loss/len(val_loader)
    accuracy=correct/total
    return avg_val_loss,accuracy,classification_rep,precision,recall

def main():
    setup_experiment("Social Media Fraud Detection - Model Evaluation")
    
    with mlflow.start_run():
        # Evaluate Text Model
        logger.info("Starting text model evaluation...")
        val_loss,accuracy,classification_rep,precision,recall=evaluate_model(model,criterion,val_loader,device)
        print(f"Validation Loss: {val_loss}, Accuracy: {accuracy}")
        print("Classification Report:")
        print(classification_rep)
        logger.info("Text model evaluation completed")

        # Evaluate Image Model
        logger.info("Starting image model evaluation...")
        img_val_loss,img_accuracy,img_classification_rep,img_precision,img_recall=evaluate_image_model(image_model,criterion,image_val_loader,device)
        print(f"Image Validation Loss: {img_val_loss}, Image Accuracy: {img_accuracy}")
        print("Image Classification Report:")
        print(img_classification_rep)
        logger.info("Image model evaluation completed")

if __name__=="__main__":
    main()