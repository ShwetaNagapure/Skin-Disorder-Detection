
# Skin Disorder Detection using CNN & Explainable AI

An AI-powered diagnostic system that classifies various skin conditions using dermatoscopic images. This project uses Deep Learning (CNNs) and Explainable AI (XAI) to help dermatologists and general practitioners diagnose skin disorders accurately and transparently.

---

## Table of Contents

* [Introduction](#introduction)
* [Motivation](#motivation)
* [Problem Statement](#problem-statement)
* [Objectives](#objectives)
* [System Architecture](#system-architecture)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Explainability](#explainability)
* [Chatbot Integration](#chatbot-integration)
* [Results](#results)
* [Installation](#installation)
* [Future Scope](#future-scope)


---

## Introduction

Skin diseases such as melanoma, basal cell carcinoma, and benign keratosis are increasingly common and often hard to diagnose visually. This project automates the diagnosis using CNNs trained on dermatoscopic images from the HAM10000 dataset, providing high classification accuracy and interpretation via XAI techniques.

---

## Motivation

* Rising global incidence of skin cancer.
* Lack of specialized dermatologists in rural and underdeveloped regions.
* Manual diagnostics are time-consuming and error-prone.
* Need for interpretable AI to build trust in automated systems.

---

## Problem Statement

Manual diagnosis is subjective and inconsistent. Existing computer vision solutions require extensive feature engineering and lack transparency. There's a need for a scalable, interpretable AI model that can generalize across varied skin types and lesions.

---

## Objectives

* Develop a CNN-based multi-class classifier for skin disease detection.
* Utilize the HAM10000 dataset for diverse training samples.
* Apply data balancing and augmentation to ensure robust learning.
* Integrate Grad-CAM and SHAP for explainability.
* Support patient interaction via a RAG-based chatbot.

---

## System Architecture

```
+----------------+       +-------------------+       +------------------+
|  User Uploads  |  -->  |  CNN Classification|  -->  |  XAI Heatmaps    |
+----------------+       +-------------------+       +------------------+
        ↓                         ↓                            ↓
   RAG Chatbot             Report Generation               JSON / UI Output
```

* Preprocessing: OpenCV, NumPy, TensorFlow/Keras
* CNN: Custom architecture using Keras
* Explainability: Grad-CAM, SHAP
* Chatbot: LangChain-based RAG (Retrieval-Augmented Generation)

---

## Dataset

* Source: [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
* Classes:

  * Actinic keratoses (akiec)
  * Basal cell carcinoma (bcc)
  * Benign keratosis (bkl)
  * Dermatofibroma (df)
  * Melanoma (mel)
  * Melanocytic nevi (nv)
  * Vascular lesions (vasc)

---

## Model Architecture

* Input: 32×32 RGB image
* 3 Convolutional blocks with:

  * 256 → 128 → 64 filters
  * BatchNorm, MaxPooling, Dropout
* Fully connected dense layer: 32 neurons
* Output layer: 7 neurons with Softmax
* Trained for 50 epochs with Adam optimizer and categorical\_crossentropy loss.

---

## Explainability

* Grad-CAM: Highlights key regions influencing the CNN's decision.
* SHAP (Optional): For feature-level interpretability.
* Outputs visual overlays for each prediction.

---

## Chatbot Integration

* Based on LangChain RAG architecture
* Provides treatment guidance, prediction explanations, and references to dermatological sources.
* Designed to help non-specialist users interpret results.

---

## Results

| Class | Precision | Recall | F1-Score |
| ----- | --------- | ------ | -------- |
| akiec | 0.81      | 0.90   | 0.85     |
| bcc   | 0.81      | 0.78   | 0.80     |
| bkl   | 0.74      | 0.71   | 0.73     |
| df    | 0.97      | 0.98   | 0.98     |
| mel   | 0.74      | 0.62   | 0.67     |
| nv    | 0.71      | 0.76   | 0.73     |
| vasc  | 0.95      | 1.00   | 0.98     |

* Overall Accuracy: 82.17%
* Macro F1-Score: 0.82

---

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/skin-disorder-detection.git
cd skin-disorder-detection

# Install dependencies
pip install -r requirements.txt

# Start training
python src/train.py

# Predict
python src/predict.py --image path/to/image.jpg
```

---

## Future Scope

* Integrate with clinical systems for real-time diagnostics.
* Improve minority class performance via GAN-generated samples.
* Extend chatbot functionality with voice interaction.
* Support multi-modal input (text + image).

