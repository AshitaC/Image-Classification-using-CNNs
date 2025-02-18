# CIFAR10 Image-Classification-using-CNNs

# Overview
This project implements a **Convolutional Neural Network (CNN) using TensorFlow and Keras** to classify images from the **CIFAR-10 dataset** into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The model was optimized by experimenting with different **network architectures, activation functions (ReLU, Swish), dropout, batch normalization, and hyperparameters** to enhance accuracy while mitigating overfitting.

# Project Highlights

- **Dataset**: CIFAR-10 (60,000 images, 10 classes)
- **Model**: 5-layer CNN with batch normalization, dropout, and L2 regularization
- **Technologies Used**: TensorFlow, Keras, Python
- **Training**: Executed on BSU R2 cluster and Google Colab (GPU-enabled)
- **Final Accuracy**: 85.21% on test data

# Architecture & Methodology

## Data Preprocessing:

Normalized pixel values between 0 and 1 for stable training.

Split dataset into training (45,000), validation (5,000), and test (10,000) images.

## CNN Model Design:
- **5 Convolutional Layers** (3×3 filters, ReLU activation).
- **Max Pooling** after selected layers to retain essential features.
- **Dropout (0.2) & L2 Regularization (0.00001)** to prevent overfitting.
- **Batch Normalization** to stabilize training.
- **Fully Connected Layers**: Two dense layers (1024 neurons each) followed by a softmax output layer.

## Optimization & Training:
- **Adam optimizer** with **categorical cross-entropy loss**.
- **Batch size**: 32, **Epochs**: 25.
- **Hyperparameter tuning**: Learning rate adjustments (0.0001 optimal).

# Results

- Training Accuracy	95.03%
- Validation Accuracy	87.12%
- Test Accuracy	85.21%

# Future Improvements
- Implement **Transfer Learning** with pre-trained models (ResNet, VGG).
- Explore **attention mechanisms** to enhance feature extraction.
