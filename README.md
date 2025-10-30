# AI-Radiology-Practice-Platform
# Breast Ultrasound AI Tutor

## Overview
This is an interactive Streamlit application designed to help medical students and radiology residents practice interpreting breast ultrasound images. It uses an AI model for segmentation and classification of breast lesions, providing educational feedback through a virtual tutor named "Dr. Nova".

The app allows users to:
- Select demo ultrasound cases
- Draw bounding boxes around suspicious lesions
- Classify lesions as normal, benign, or malignant
- Document clinical reasoning
- Chat with an AI tutor for guidance
- Compare their assessment with AI analysis (segmentation, classification, Grad-CAM attention maps)
- Receive structured feedback on performance

## Features
- **Drawable Canvas:** Users can draw regions of interest on ultrasound images.
- **AI Model:** Uses a custom YOLO-multitask model based on ResNet34 for classification and segmentation.
- **Grad-CAM Visualization:** Shows AI attention maps for interpretability.
- **Performance Metrics:** Calculates Dice score for segmentation overlap and compares classifications.
- **AI Tutor:** Powered by Groq (optional) for conversational guidance and feedback.
- **Educational Resources:** Sidebar with BI-RADS reference, key features, and learning objectives.

## Requirements
- Python 3.8+
- Streamlit
- PyTorch
- Torchvision
- TorchCAM
- Pillow (PIL)
- NumPy
- Matplotlib
- Groq (optional for AI tutor)
- dotenv (for environment variables)

Install dependencies:
