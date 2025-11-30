# HandMatrix: Gesture Recognition Engine ✋🤖

A comprehensive Computer Vision project implemented in C++ and Python to recognize hand gestures in real-time. This project explores and compares two distinct approaches: a **Custom Neural Network (MLP)** built entirely from scratch in C++ and a modern **Convolutional Neural Network (CNN)** trained in TensorFlow and deployed via ONNX.

![C++](https://img.shields.io/badge/C++-20-blue.svg) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg) ![Python](https://img.shields.io/badge/Python-3.12-yellow.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)

## 📂 Project Overview

This repository is structured into four distinct modules:

1.  **Data Collector (C++):** A specialized tool to capture, pre-process, and label grayscale hand gestures into organized subfolders.
2.  **Data Augmentation & Training (Python):** Scripts to augment raw data (rotation, noise) and train a CNN using TensorFlow/Keras, exporting the result to ONNX.
3.  **Custom NN Inference (C++):** A Multilayer Perceptron implemented purely in C++ using the Eigen3 library, featuring manual Backpropagation.
4.  **CNN Inference (C++):** Live inference engine using the Python-trained model via OpenCV's DNN module.

---

## 👁️ The Computer Vision Pipeline

Before any Neural Network can classify a gesture, the raw webcam image must be heavily processed to extract relevant features and reduce noise. Both models use the exact same preprocessing pipeline to ensure consistency.

### 1. Region of Interest (ROI) & Background Subtraction
We focus only on a specific area (`scanBox`) to reduce computational load. To isolate the hand from the background, we calculate the absolute difference between the current frame and a stored background frame.

```cpp
// Calculate difference between current frame (cleanRoi) and stored background
cv::absdiff(cleanRoi, background, diff);

// Create a binary mask: Pixels with high difference become white (255), others black (0)
cv::threshold(diff, mask, 30, 255, cv::THRESH_BINARY);
