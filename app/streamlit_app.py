import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subprocess
import time

import streamlit as st
import requests
import json
from PIL import Image
import io
from src.artifacts import download_model_artifacts

download_model_artifacts()



if "api_started" not in st.session_state:
    subprocess.Popen([sys.executable,"-m","uvicorn","app.api:app","--host","0.0.0.0","--port", "8000"])
    time.sleep(5)  # wait for the server to start
    st.session_state.api_started = True

# Configure page
st.set_page_config(
    page_title="Social Media Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title(" Social Media Fraud Detection System")
st.markdown("---")

# API base URL
API_BASE_URL = "http://localhost:8000"

# Helper function to format predictions
def format_predictions(probs, preds, mode):
    """Format prediction results for display"""
    results = []
    for i, (prob, pred) in enumerate(zip(probs, preds)):
        prediction_label = " FAKE" if pred == 1 else " REAL"
        confidence = max(prob) * 100 if prob else 0
        results.append({
            "index": i + 1,
            "prediction": prediction_label,
            "confidence": confidence,
            "probabilities": prob
        })
    return results

# Helper function to display results
def display_results(response_data):
    """Display prediction results in a nice format"""
    probs = response_data.get("probs", [])
    preds = response_data.get("preds", [])
    mode = response_data.get("mode", "unknown")
    
    st.success("✅ Prediction completed!")
    
    # Display mode
    mode_display = {
        "text_only": " Text Analysis",
        "image_only": " Image Analysis",
        "text_image": " Fusion (Text + Image) Analysis"
    }
    st.info(f"**Analysis Mode:** {mode_display.get(mode, mode)}")
    
    # Display results
    results = format_predictions(probs, preds, mode)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predictions")
        for result in results:
            with st.container():
                st.markdown(f"""
                <div class='metric-box'>
                    <strong>Item {result['index']}</strong><br>
                    Prediction: {result['prediction']}<br>
                    Confidence: {result['confidence']:.2f}%
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Probabilities")
        for i, result in enumerate(results):
            if result["probabilities"]:
                col1_prob, col2_prob = st.columns(2)
                with col1_prob:
                    st.metric(f"Item {result['index']} - Real", f"{result['probabilities'][0]:.4f}")
                with col2_prob:
                    st.metric(f"Item {result['index']} - Fake", f"{result['probabilities'][1]:.4f}")


# Create tabs
tab1, tab2, tab3 = st.tabs([" Text Analysis", " Image Analysis", " Fusion Analysis"])

with tab1:
    st.header("Text-Only Fraud Detection")
    st.markdown("Analyze text content for potential fraud indicators.")
    
    # Input method
    input_method = st.radio(
        "How would you like to input text?",
        ["Single Text Entry", "Multiple Texts", "Paste CSV"],
        key="text_input_method"
    )
    
    text_inputs = []
    
    if input_method == "Single Text Entry":
        text = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Paste your social media post, comment, or message here..."
        )
        if text.strip():
            text_inputs = [text]
    
    elif input_method == "Multiple Texts":
        num_texts = st.slider("How many texts do you want to analyze?", 1, 5, 2)
        for i in range(num_texts):
            text = st.text_area(
                f"Text {i+1}:",
                height=100,
                placeholder=f"Enter text {i+1}...",
                key=f"text_{i}"
            )
            if text.strip():
                text_inputs.append(text)
    
    else:  # Paste CSV
        csv_text = st.text_area(
            "Paste comma-separated texts (one per line):",
            height=150,
            placeholder="Text 1\nText 2\nText 3..."
        )
        if csv_text.strip():
            text_inputs = [line.strip() for line in csv_text.split('\n') if line.strip()]
    
    if st.button(" Analyze Text", key="analyze_text", type="primary"):
        if not text_inputs:
            st.error(" Please enter at least one text to analyze.")
        else:
            with st.spinner(" Analyzing text..."):
                try:
                    # Prepare request
                    files = [("text_input", (None, text)) for text in text_inputs]
                    response = requests.post(
                        f"{API_BASE_URL}/predict/text",
                        data={"text_input": text_inputs}
                    )
                    
                    if response.status_code == 200:
                        display_results(response.json())
                    else:
                        st.error(f"API Error: {response.status_code}")
                        st.json(response.json())
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Make sure the API server is running at http://localhost:8000")


with tab2:
    st.header("Image-Only Fraud Detection")
    st.markdown("Analyze images for potential fraud indicators.")
    
    uploaded_files = st.file_uploader(
        "Upload one or more images:",
        type=["jpg", "jpeg", "png", "bmp", "gif"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f" Uploaded Files ({len(uploaded_files)})")
            for idx, file in enumerate(uploaded_files):
                st.write(f"{idx+1}. {file.name} ({file.size} bytes)")
        
        with col2:
            st.subheader(" Preview")
            selected_idx = st.selectbox(
                "Select image to preview:",
                range(len(uploaded_files)),
                format_func=lambda x: uploaded_files[x].name
            )
            image = Image.open(uploaded_files[selected_idx])
            st.image(image, use_column_width=True, caption=uploaded_files[selected_idx].name)
    
    if st.button(" Analyze Images", key="analyze_images", type="primary"):
        if not uploaded_files:
            st.error(" Please upload at least one image to analyze.")
        else:
            with st.spinner(" Analyzing images..."):
                try:
                    # Prepare files for multipart upload
                    files = [("images", (file.name, file.getbuffer(), file.type)) for file in uploaded_files]
                    response = requests.post(
                        f"{API_BASE_URL}/predict/image",
                        files=files
                    )
                    
                    if response.status_code == 200:
                        display_results(response.json())
                    else:
                        st.error(f" API Error: {response.status_code}")
                        st.json(response.json())
                except Exception as e:
                    st.error(f" Error: {str(e)}")
                    st.info("Make sure the API server is running at http://localhost:8000")


with tab3:
    st.header("Fusion Analysis (Text + Image)")
    st.markdown("Analyze both text and images together for more accurate fraud detection.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Text Input")
        num_texts_fusion=1
        fusion_texts = []
        for i in range(num_texts_fusion):
            text = st.text_area(
                f"Text {i+1}:",
                height=80,
                placeholder=f"Enter text {i+1}...",
                key=f"fusion_text_{i}"
            )
            if text.strip():
                fusion_texts.append(text)
    
    with col2:
        st.subheader("Image Upload")
        fusion_images = st.file_uploader(
            "Upload images:",
            type=["jpg", "jpeg", "png", "bmp", "gif"],
            accept_multiple_files=True,
            key="fusion_images"
        )
        
        if fusion_images:
            st.write(f"**Files uploaded:** {len(fusion_images)}")
            for idx, file in enumerate(fusion_images):
                st.write(f"  {idx+1}. {file.name}")
    
    if st.button(" Analyze Fusion", key="analyze_fusion", type="primary"):
        if not fusion_texts or not fusion_images:
            st.error(" Please provide both text and images for fusion analysis.")
        else:
            with st.spinner("Analyzing text and images..."):
                try:
                    # Prepare data
                    data = {"text_input": fusion_texts}
                    files = [("images", (file.name, file.getbuffer(), file.type)) for file in fusion_images]
                    
                    response = requests.post(
                        f"{API_BASE_URL}/predict/fusion",
                        data=data,
                        files=files
                    )
                    
                    if response.status_code == 200:
                        display_results(response.json())
                    else:
                        st.error(f" API Error: {response.status_code}")
                        st.json(response.json())
                except Exception as e:
                    st.error(f" Error: {str(e)}")
                    st.info("Make sure the API server is running at http://localhost:8000")

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.info(" API Status: Check if server is running at http://localhost:8000")
with col2:
    st.info(" API Docs: Visit http://localhost:8000/docs for Swagger UI")
with col3:
    st.info(" OpenAPI Schema: Visit http://localhost:8000/openapi.json")