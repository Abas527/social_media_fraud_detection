## src/data_loader.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import split_data
import numpy as np
import torch
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader, Dataset
from src.preprocessing import transformer
from PIL import Image
import os
from pathlib import Path

tokenizer=DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize_data(texts):
    if not isinstance(texts, list):
        texts=texts.tolist()

        

    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

class TextDataset(Dataset):
    def __init__(self,embeddings,labels):
        self.embeddings=embeddings
        self.labels=torch.tensor(labels.values)
    
    def __getitem__(self,index):
        item={key:val[index] for key,val in self.embeddings.items()}
        item['labels']=self.labels[index]
        return item,self.labels[index]
    def __len__(self):
        return len(self.labels)

def create_embedding(df):
    #step 1: split data
    train_df,val_df,test_df=split_data(df)
    
    #step 2: tokenize data
    train_embedding=tokenize_data(train_df['texts'])
    val_embedding=tokenize_data(val_df['texts'])
    test_embedding=tokenize_data(test_df['texts'])
    return train_embedding,train_df['label'],val_embedding,val_df['label'],test_embedding,test_df['label']

def create_dataloaders(df):
    #step 1: split data
    train_df,val_df,test_df=split_data(df)
    
    #step 2: tokenize data
    train_embedding=tokenize_data(train_df['texts'])
    val_embedding=tokenize_data(val_df['texts'])
    test_embedding=tokenize_data(test_df['texts'])

    #step 3:create datasets
    train_dataset=TextDataset(train_embedding,train_df['label'])
    val_dataset=TextDataset(val_embedding,val_df['label'])
    test_dataset=TextDataset(test_embedding,test_df['label'])

    #step 4:create dataloaders
    train_dataloader=DataLoader(train_dataset,batch_size=16,shuffle=True)
    val_dataloader=DataLoader(val_dataset,batch_size=16,shuffle=False)
    test_dataloader=DataLoader(test_dataset,batch_size=16,shuffle=False)

    # batch_example=next(iter(train_dataloader))
    # print(batch_example["input_ids"].shape)
    # print(batch_example["attention_mask"].shape)
    # print(batch_example["labels"].shape)

    return train_dataloader,val_dataloader,test_dataloader

#####create images and labels list


def create_image_labels_list(data_dir:Path):
    images=[]
    labels=[]

    for img in os.listdir(data_dir/"fake"):
        images.append(os.path.join(data_dir/"fake", img))
        labels.append(0)
    
    for img in os.listdir(data_dir/"real"):
        images.append(os.path.join(data_dir/"real", img))
        labels.append(1)
    
    return images, pd.Series(labels)

def create_image_dataset(data_dir:Path):
    images,labels=create_image_labels_list(data_dir)
    train_img, val, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
    val_img, test_img, val_labels, test_labels = train_test_split(val, val_labels, test_size=0.5, random_state=42, stratify=val_labels)


    return train_img,train_labels,val_img,val_labels,test_img,test_labels



# image dataset

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = torch.tensor(labels.values, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {image_path}") from e

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]
        return image, label


def create_image_loader(data_dir:Path):
    images,labels=create_image_labels_list(data_dir)
    train_img, val_img, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
    train_dataset=ImageDataset(train_img,train_labels,transform=transformer())
    val_dataset=ImageDataset(val_img,val_labels,transform=transformer())

    train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=16,shuffle=False)

    return train_loader,val_loader




def combine_text_and_image(text_df, data_dir: Path, processed_path: Path):
    image_paths, labels = create_image_labels_list(data_dir)
    image_df = pd.DataFrame({
        "label": labels,
        "image_path": image_paths
    })

    combined = []

    for label in text_df["label"].unique():
        text_subset = text_df[text_df["label"] == label].reset_index(drop=True)
        image_subset = image_df[image_df["label"] == label].reset_index(drop=True)

        n = min(len(text_subset), len(image_subset))

        if n == 0:
            continue

        text_subset = text_subset.sample(n, random_state=42).reset_index(drop=True)
        image_subset = image_subset.sample(n, random_state=42).reset_index(drop=True)

        text_subset["image_path"] = image_subset["image_path"].values
        combined.append(text_subset)

    df = pd.concat(combined).sample(frac=1, random_state=42).reset_index(drop=True)

    processed_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path / "text_image_data.csv", index=False)

    return df



def main():
    # df=pd.read_csv("data/processed/text/text_data.csv")
    # train_loader,val_loader,test_loader=create_dataloaders(df)

    text_df=pd.read_csv("data/processed/text/text_data.csv")
    combine_text_and_image(text_df,Path("data/raw/image"),Path("data/processed/fusion"))

if __name__=="__main__":
    main()