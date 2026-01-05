🎭 GAN-based Face Generation with VGG16 Feature Analysis

This project implements a Generative Adversarial Network (GAN) for synthetic face generation, combined with a VGG16-based feature extractor for attribute prediction and perceptual evaluation

📌 Project Overview

The system consists of:

A Generator that creates realistic fake face images from random noise
A Discriminator that distinguishes real faces from generated ones
A VGG16 network used for:
Facial attribute classification (smile, gender, glasses, etc.)
Feature embeddings for perceptual similarity and FID evaluation

🧩 Components
🔹 Generator

Input: Random noise vector z
Output: Synthetic face images
Objective: Fool the discriminator

🔹 Discriminator
Input: Real or fake face image
Output: Probability of image being real
Objective: Correctly classify real vs fake

🔹 VGG16 (Pretrained)
Used after GAN training
Two usage modes:
Option A: Attribute prediction (Smile, Gender, Glasses)
Option B: Feature embeddings for perceptual metrics (FID)

📊 Use Cases

Face synthesis research
GAN understanding & experimentation
Facial attribute analysis
Feature-based similarity comparison
Resume & academic projects


🛠️ Tech Stack
Python
PyTorch / TensorFlow
OpenCV
NumPy
Pretrained VGG16
Matplotlib
