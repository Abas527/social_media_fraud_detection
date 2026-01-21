import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

#text data processing
RAW_DATA_DIR=Path("data/raw/")
TEXT_DIR=RAW_DATA_DIR / "text/"
PROCESS_DATA_DIR=Path("data/processed/")


def process_text_files(text_path:Path):
    fake=pd.read_csv(text_path / "Fake.csv")
    true=pd.read_csv(text_path / "True.csv")
    
    fake["label"]=0
    true["label"]=1

    df=pd.concat([fake,true]).sample(frac=1).reset_index(drop=True) # sampling and mixing the data
    df["texts"]=df["title"]+" "+df["text"]
    df["texts"]=df["texts"].str.strip("  ").str.lower()                           # combining the text and title
    df.drop(columns=["date","subject","title","text"],inplace=True)        # dropping useless columns


    return df

def process_new_text_file(text_path:Path):
    train=pd.read_csv(text_path / "train.csv", on_bad_lines='skip', encoding='ISO-8859-1')
    test=pd.read_csv(text_path / "test.csv",on_bad_lines='skip', encoding='ISO-8859-1')
    df=pd.concat([train,test]).reset_index(drop=True)
    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)
    df["texts"]=df["description"]+" "+df["subject"]+" "+df["topic"]+" "+df["context"]
    df["texts"]=df["texts"].str.strip("  ").str.lower()                           # combining the text and title

    df["label"]=df["label"].map({"half-true":0,"mostly-true":1,"barely-true":0,"FALSE":0,"pants-fire":0,"TRUE":1})

    df.drop(columns=["id","description","subject","topic","context","a","b","c","d","e","person","place","side"],inplace=True)
    
    return df

def combine_two_datasets(df1:pd.DataFrame, df2:pd.DataFrame,processed_path:Path):
    df=pd.concat([df1,df2]).sample(frac=1).reset_index(drop=True)
    processed_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path / "text_data.csv", index=False)
    return df

def split_data(df):
    train_val_df,test_df=train_test_split(df,test_size=0.15,random_state=42,stratify=df["label"])
    train,val=train_test_split(train_val_df,test_size=0.1765,random_state=42,stratify=train_val_df["label"])
    return train,val,test_df




#image data processing
from PIL import Image
from torchvision import transforms

transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

def transformer():
    return transform





def main():
    df=process_new_text_file(TEXT_DIR)
    df1=process_text_files(TEXT_DIR)
    combined_df=combine_two_datasets(df,df1,PROCESS_DATA_DIR/"text")

    train,val,test=split_data(combined_df)
    print("Train",train["label"].value_counts(normalize=True))
    print("Val",val["label"].value_counts(normalize=True))
    print("Test",test["label"].value_counts(normalize=True))


if __name__=="__main__":
    main()