# MNIST-TensorFlow-codes

A simple Convolutional Neural Network (CNN) built from scratch using TensorFlow/Keras to classify handwritten digits from the MNIST dataset.

🚀 Project Overview
This project demonstrates a basic CNN architecture applied to the MNIST dataset (28x28 grayscale images of digits 0–9). The goal was to explore CNNs, understand training dynamics, and achieve high validation accuracy in a straightforward way.

Key takeaways:
Normalizing pixel values speeds up convergence.
Even a small CNN can achieve ~98.5% validation accuracy.
Simple visualizations make it fun to see predictions vs actual digits.

🛠️ Features

Conv2D layers for feature extraction
MaxPooling2D layers for downsampling
Dense layers for classification
Softmax activation for output probabilities
Uses sparse_categorical_crossentropy for loss
Validation split to monitor generalization

📈 Results

Training Accuracy: ~99.7%
Validation Accuracy: ~98.5%
Loss decreases steadily, no major overfitting observed
