
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from PIL import Image

from src.model import TextModel, ImageModel
from transformers import AutoTokenizer

import mlflow
from src.mlflow_utils import setup_experiment
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

setup_experiment("Fusion_Text_Image_Classification")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_model = TextModel().to(device)
image_model = ImageModel().to(device)

text_model.load_state_dict(torch.load("models/text_model.pth", map_location=device, weights_only=True))
image_model.load_state_dict(torch.load("models/image_model.pth", map_location=device, weights_only=True))

# Freeze pretrained models
text_model.eval()
image_model.eval()
for param in text_model.parameters():
    param.requires_grad = False
for param in image_model.parameters():
    param.requires_grad = False

# define dataset
class FusionDataset(Dataset):
    def __init__(self, df, tokenizer, image_transform=None, max_len=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        encoding = self.tokenizer(
            row["texts"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        text_inputs = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

        image = Image.open(row["image_path"]).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)

        label = torch.tensor(row["label"], dtype=torch.long)

        return text_inputs, image, label

#define fusion model
class FusionModel(nn.Module):
    def __init__(self, text_dim=768, image_dim=2048, num_classes=2):
        super(FusionModel, self).__init__()
        self.text_model = text_model
        self.image_model = image_model

        self.classifier = nn.Sequential(
            nn.Linear(text_dim + image_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, text_features, image_features):
        fused = torch.cat((text_features, image_features), dim=1)
        return self.classifier(fused)

def create_fusion_dataloader(df, tokenizer, image_transform=None, batch_size=16, shuffle=True):
    dataset = FusionDataset(df, tokenizer, image_transform=image_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_fusion_model(model, optimizer, criterion, train_loader, device, epochs=5):
    with mlflow.start_run(run_name="fusion_model_training"):
        # Log hyperparameters
        mlflow.log_param("model_type", "FusionModel")
        mlflow.log_param("base_models", "TextModel + ImageModel")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", optimizer.defaults['lr'])
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("optimizer", "AdamW")
        mlflow.log_param("loss_function", "CrossEntropyLoss")
        mlflow.log_param("text_dim", 768)
        mlflow.log_param("image_dim", 2048)
        
        logger.info("Starting fusion model training...")
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for text_batch, images, labels in train_loader:
                labels = labels.to(device)
                images = images.to(device)
                text_batch = {k: v.to(device) for k, v in text_batch.items()}

                with torch.no_grad():
                    _, text_features = text_model(**text_batch)
                    _, image_features = image_model(images)

                outputs = model(text_features, image_features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Log metrics per epoch
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        logger.info("Fusion model training completed")

    torch.save(model.state_dict(), "models/fusion_model.pth")

def evaluate_fusion_model(model, criterion, val_loader, device):
    with mlflow.start_run(run_name="fusion_model_evaluation"):
        logger.info("Starting fusion model evaluation...")
        
        model.eval()
        all_labels, all_preds = [], []

        with torch.no_grad():
            for text_batch, images, labels in val_loader:
                labels = labels.to(device)
                images = images.to(device)
                text_batch = {k: v.to(device) for k, v in text_batch.items()}

                _, text_features = text_model(**text_batch)
                _, image_features = image_model(images)

                outputs = model(text_features, image_features)
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        classification_rep = classification_report(all_labels, all_preds, output_dict=True)
        confusion_mat = confusion_matrix(all_labels, all_preds)
        
        print("Classification Report:")
        print(classification_report(all_labels, all_preds))
        print("Confusion Matrix:")
        print(confusion_mat)
        
        # Log metrics
        mlflow.log_metric("accuracy", classification_rep['accuracy'])
        mlflow.log_metric("precision", classification_rep['weighted avg']['precision'])
        mlflow.log_metric("recall", classification_rep['weighted avg']['recall'])
        mlflow.log_metric("f1_score", classification_rep['weighted avg']['f1-score'])
        
        # Log classification report as text artifact
        mlflow.log_text(classification_report(all_labels, all_preds), "fusion_classification_report.txt")
        
        logger.info("Fusion model evaluation completed and logged")

def main():

    text_image_df = pd.read_csv("data/processed/fusion/text_image_data.csv")

    train_df, val_df = train_test_split(
        text_image_df,
        test_size=0.2,
        random_state=42,
        stratify=text_image_df['label']
    )


    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # DataLoaders
    train_loader = create_fusion_dataloader(train_df, tokenizer, image_transform, batch_size=16, shuffle=True)
    val_loader = create_fusion_dataloader(val_df, tokenizer, image_transform, batch_size=16, shuffle=False)

    print("Train batches:", len(train_loader), "Val batches:", len(val_loader))

    # Fusion model
    fusion_model = FusionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=2e-5)

    # Train & Evaluate
    logger.info("Starting fusion model training and evaluation pipeline...")
    train_fusion_model(fusion_model, optimizer, criterion, train_loader, device, epochs=5)
    evaluate_fusion_model(fusion_model, criterion, val_loader, device)
    logger.info("Fusion model pipeline completed")

if __name__ == "__main__":
    main()
