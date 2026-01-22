#!/bin/bash

echo "Starting container..."

echo "Downloading models..."
python src/artifacts.py

echo "Starting FastAPI..."
uvicorn app.api:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit..."
streamlit run app/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0
