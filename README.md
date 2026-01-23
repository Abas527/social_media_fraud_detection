# Social Media Fraud Detection

### Multimodal Text–Image Classification System

---

## Table of Contents

* [Overview](#overview)
* [Problem Statement](#problem-statement)
* [Solution Overview](#solution-overview)
* [System Architecture](#system-architecture)
* [Model Architecture](#model-architecture)
* [Fusion Strategy](#fusion-strategy)
* [Inference Logic](#inference-logic)
* [Application Design](#application-design)
* [Project Structure](#project-structure)
* [Model Outputs](#model-outputs)
* [Deployment Strategy](#deployment-strategy)
* [Design Decisions](#design-decisions)
* [Limitations](#limitations)
* [Future Improvements](#future-improvements)
* [Conclusion](#conclusion)

---

## Overview

Social media platforms are increasingly exploited for fraudulent activities such as phishing, impersonation, fake giveaways, and scam promotions. These frauds often combine misleading text with manipulative visual content, making unimodal detection systems insufficient.

This project presents a production-oriented multimodal fraud detection system that analyzes text, images, or both to classify social media content as fraudulent or legitimate. The system is designed with real-world constraints in mind, including partial data availability, inference efficiency, and deployment readiness.

The application exposes a simple user interface that allows users to submit text, images, or a combination of both and receive a fraud prediction in real time.

The system is designed with **production inference in mind**, focusing on modularity, scalability, and robustness.

---

## Problem Statement

Social media fraud often manifests as:

* Fake giveaways or investment scams
* Threatening or manipulative messages
* Misleading screenshots or fabricated images
* Coordinated use of text and visuals to increase credibility

The key challenges:

1. Input modalities may be incomplete or missing
2. Fraud signals may exist in only one modality
3. Models must remain efficient for real-time inference

---

## Solution Overview

We propose a **modular multimodal learning system** with **late fusion**, allowing each modality to be processed independently while enabling joint reasoning when both inputs are present.

The system dynamically selects the best inference path:

* Text Model → when only text is available
* Image Model → when only an image is available
* Fusion Model → when both modalities are present

---

## System Architecture

```
User Input
   ├── Text ───────────────► Text Model ─────┐
   │                                          │
   ├── Image ──────────────► Image Model ─────┤
   │                                          │
   └── Text + Image ───────► Fusion Model ◄───┘
                                 │
                            Final Prediction
```

---

## Model Architecture

### Text Classification Model

* **Backbone:** BERT (bert-base-uncased)
* **Input:** Raw social media text
* **Output:**

  * Classification logits
  * Semantic embedding (768-dimensional)

The text model captures linguistic cues such as urgency, persuasion, threats, and deceptive language patterns.

---

### Image Classification Model

* **Backbone:** ResNet-50
* **Input:** Uploaded image
* **Output:**

  * Classification logits
  * Visual embedding (2048-dimensional)

The image model identifies visual artifacts such as misleading screenshots, fake proof images, or suspicious patterns.

---

### Fusion Model

* **Fusion Type:** Late fusion
* **Input:** Concatenated text and image embeddings
* **Architecture:** Fully connected classifier
* **Output:** Final fraud classification

---

## Fusion Strategy

Late fusion is chosen over early fusion due to:

* Independent training of text and image models
* Robustness when one modality is noisy or missing
* Easier debugging and scalability
* Better real-world performance in asynchronous data scenarios

```
Text Embedding (768)
        │
        ├── Concatenation ──► MLP Classifier ──► Fraud / Non-Fraud
        │
Image Embedding (2048)
```

---

## Inference Logic

The system automatically selects the inference strategy based on input availability:

| Input Type   | Model Used   |
| ------------ | ------------ |
| Text only    | Text Model   |
| Image only   | Image Model  |
| Text + Image | Fusion Model |

This avoids forcing empty inputs and ensures stable predictions.

---

## Application Design

### Frontend

* Simple and intuitive user interface
* Text input field
* Image upload support
* Clear prediction results with confidence scores

### Backend Logic

* CPU-compatible inference
* Lazy model loading
* Unified prediction output format
* No training logic in production

---

## Project Structure

```
.
├── app/
│   ├── api.py              # Prediction API logic
│   ├── streamlit_app.py    # UI interface
│
├── src/
│   ├── data_loader.py      # Define the Dataset and loader
│   ├── models.py           # Text, Image, Fusion models
│   ├── predict.py        # Prediction routing logic
│   ├── preprocessing.py   # Tokenization & transforms
|   |── train.py          # training the text and image model
│   ├── evaluate.py       # Evaluating the text and image model
│   ├── fusion_model.py   #training and evaluation of fusion model    
│   ├── mlflow_utils.py   # mlflow setup
├── models/                 # Trained model weights
├── requirements.txt
├── README.md
```

---

## Model Outputs

All prediction endpoints return a unified response format:

```json
{
  "mode": "text_only | image_only | fusion",
  "preds": [0],
  "probs": [[0.74, 0.26]]
}
```

* `mode`: Inference path used
* `preds`: Predicted class label
* `probs`: Class probabilities

---

## Deployment Strategy

* Inference-only deployment
* CPU-based runtime
* No training or fine-tuning in production
* Optimized for low latency and reliability

The architecture supports easy migration to:

* Cloud APIs
* Batch processing pipelines
* Content moderation systems

* currently we are on huggingface space
https://huggingface.co/spaces/cyberanil/fraud_detection

You can visit and enjoy the app
---

## Design Decisions

* **Late fusion** for robustness and flexibility
* **Pretrained models** for faster convergence and better generalization
* **Modular architecture** for maintainability
* **Single unified inference interface** for simplicity

---

## Limitations

* Does not currently provide explainability visualizations
* Limited to binary classification
* Performance depends on quality of pretrained weights

---

## Future Improvements

* Explainable AI (Grad-CAM, attention maps)
* Multi-class fraud taxonomy
* Batch inference support
* Active learning feedback loop
* Dedicated REST API deployment

---

## Conclusion

This project demonstrates a **real-world multimodal machine learning system** built with production considerations in mind. By combining strong pretrained models with a flexible fusion strategy, the system effectively detects social media fraud across diverse input scenarios.


