import os
import gdown
import subprocess

DRIVE_URL="https://drive.google.com/drive/folders/1a1W9EPtDYwVBXm_oZWjMw4tLkr2duZzG?usp=sharing"

def download_model_artifacts():

    if not os.path.exists("models"):
        os.makedirs("models")
        subprocess.run(["gdown", "--folder", DRIVE_URL, "-O", "models"])
    else:
        print("Artifacts already downloaded.")

if __name__ == "__main__":
    download_model_artifacts()