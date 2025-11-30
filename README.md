# HandMatrix: Gesture Recognition Engine ✋🤖

A comprehensive Computer Vision project implemented in C++ and Python to recognize hand gestures in real-time. This project explores two different approaches: a **Custom Neural Network (MLP)** built from scratch in C++ and a **Convolutional Neural Network (CNN)** trained in Python and deployed in C++.

![C++](https://img.shields.io/badge/C++-20-blue.svg) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg) ![Python](https://img.shields.io/badge/Python-3.12-yellow.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)

## 📂 Project Overview

This repository contains four distinct modules:
1.  **Data Collector (C++):** A tool to capture and label grayscale hand gestures.
2.  **Data Augmentation & Training (Python):** Scripts to augment data and train a CNN (exporting to ONNX).
3.  **Custom NN Inference (C++):** A Multilayer Perceptron implemented purely in C++ (Eigen library) with manual backpropagation.
4.  **CNN Inference (C++):** Live inference using the Python-trained model via OpenCV DNN module.

---

## 🧠 Theoretical Background

This project implements and compares two distinct approaches to machine learning for computer vision: a classical Multilayer Perceptron (MLP) built from scratch and a modern Convolutional Neural Network (CNN).

### 1. Multilayer Perceptron (Custom C++ Implementation)

The custom neural network is a "Feedforward Neural Network" built using matrix operations via the **Eigen3** library. It processes the input image as a flattened vector of pixel intensities.

#### Architecture Topology
* **Input Layer:** 1024 neurons ($32 \times 32$ pixels flattened).
* **Hidden Layer 1:** 256 neurons.
* **Hidden Layer 2:** 64 neurons.
* **Output Layer:** 8 neurons (representing the 8 gesture classes).

#### The Mathematical Model
Each neuron performs a weighted sum of its inputs, adds a bias, and passes the result through a non-linear activation function. For a single layer $l$, the computation is:

$$Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = \sigma(Z^{[l]})$$

Where:
* $W^{[l]}$ is the Weight Matrix connecting layer $l-1$ to $l$.
* $A^{[l-1]}$ is the Activation vector from the previous layer.
* $b^{[l]}$ is the Bias Vector.
* $\sigma$ is the Activation Function.

#### Activation Function: Sigmoid
In the custom C++ implementation, the **Sigmoid** function is used for all layers. It introduces non-linearity, allowing the network to learn complex patterns, and squashes the output between 0 and 1.

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

The derivative of the sigmoid function, which is crucial for the learning process (Backpropagation), has a convenient property:
$$\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$$

#### Learning Algorithm: Backpropagation
The network learns by minimizing a Cost Function (typically Mean Squared Error for simple implementations). We use **Gradient Descent** to update weights. The gradient is calculated using the **Chain Rule**:

To find how much a specific weight $w$ contributes to the error $C$, we calculate:

$$\frac{\partial C}{\partial w} = \underbrace{\frac{\partial C}{\partial a}}_{\text{Error from next layer}} \cdot \underbrace{\frac{\partial a}{\partial z}}_{\text{Activation derivative}} \cdot \underbrace{\frac{\partial z}{\partial w}}_{\text{Input from prev. layer}}$$

The weights are then updated using a learning rate $\eta$:
$$W_{new} = W_{old} - \eta \cdot \frac{\partial C}{\partial W}$$

---

### 2. Convolutional Neural Network (CNN)

While the MLP treats the image as an unstructured list of numbers, the CNN preserves the spatial structure (height, width, channels). This makes it translation invariant—it recognizes a hand whether it's in the top-left or bottom-right corner.

#### The Convolution Operation
Instead of fully connected weights, the CNN uses learnable **filters (kernels)**. A kernel slides over the input image, performing element-wise multiplication and summation to produce a feature map.

$$(I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m, n)$$

This allows the network to automatically detect low-level features (edges, lines) in early layers and high-level features (shapes, hand structures) in deeper layers.

#### Activation Function: ReLU
The CNN (trained in TensorFlow) typically uses the **Rectified Linear Unit (ReLU)** for hidden layers. It is computationally efficient and solves the "vanishing gradient" problem often found in deep networks using Sigmoid.

$$f(x) = \max(0, x)$$

#### Pooling (Downsampling)
To reduce computational cost and make the model robust to small spatial variations, **Max Pooling** is used. It takes a window (e.g., $2 \times 2$) and keeps only the maximum value, discarding the rest.

#### Output: Softmax
The final layer of the CNN uses the **Softmax** function to convert raw output scores (logits) into a probability distribution over the 8 classes.

$$P(y=j \mid x) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$$

This ensures that the sum of all output probabilities equals 1 (100%).
---

## 🚀 How to Use (Workflow)

To reproduce the results or train with your own data, follow this strict order:

### Step 1: Data Collection 📸
* Go to `01_Data_Collector`.
* Compile and run the C++ program.
* Press **'B'** to capture the background (ensure no hand is in the frame).
* Select a class (0-7) using number keys.
* Hold **SPACE** to record images. They are saved into labeled subfolders automatically.

### Step 2: Augmentation & Training 🐍
* Go to `02_Data_Augmenter`.
* Run `augment_data.py` to generate variations (rotations, noise) of your raw data.
* Run `train_cnn.py`. This uses TensorFlow/Keras to train the CNN.
* **Result:** A file named `hand_cnn.onnx` will be generated.

### Step 3: Deployment (Inference) ⚡

**Option A: The CNN (Recommended)**
* Go to `04_Model_CNN_Inference`.
* Copy the `hand_cnn.onnx` file into the build directory (next to the executable).
* Run the program. It uses OpenCV's DNN module to load the ONNX file.

**Option B: The Custom NN**
* Go to `03_Model_CustomNN`.
* This module reads the augmented data directly from the folders and performs training in C++ (Backpropagation) before switching to live inference mode.

---

## 🛠 Dependencies

* **C++:** OpenCV 4.x, Eigen3 (for Custom NN)
* **Python:** TensorFlow, NumPy < 2.0, tf2onnx, OpenCV-Python, tf_keras

## 📝 License
This project is open-source. Feel free to use and modify.
