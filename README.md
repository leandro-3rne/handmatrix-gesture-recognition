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
```

### 2. Feature Extraction (Edge Detection)
We use the **Canny Algorithm** to detect edges. To make the detection robust against lighting changes, we first convert the image to the HSV color space and extract the Saturation channel.

```cpp
// Convert to HSV and extract Saturation channel (channel 1)
cv::cvtColor(processingImg, hsv, cv::COLOR_BGR2HSV);
cv::split(hsv, channels);
cv::Mat saturation = channels[1];

// Apply Canny Edge Detection
cv::Canny(saturation, edges, 50, 150);
```

### 3. Morphological Operations
Raw edges are often thin and disconnected. We use **Dilation** to thicken the lines, making the shape clearer for the AI.

```cpp
// Dilate (thicken) the edges using a 3x3 kernel
cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
cv::dilate(edges, edges, kernel);
```

### 4. Downsampling & Normalization
Finally, the image is resized to a standardized $32 \times 32$ grid. For the neural network, pixel intensity values ($0-255$) are normalized to a range of $0.0-1.0$.

```cpp
// Resize to target size
cv::resize(edges, brainInput, cv::Size(32, 32));

// Normalize pixel values (0-255 -> 0.0-1.0)
// For CNN, we reshape this to (1, 32, 32, 1) later.
brainInput.convertTo(normalizedInput, CV_32F, 1.0 / 255.0);
```

---

## ⚖️ Model Comparison: MLP vs. CNN

This project highlights the fundamental differences between classical fully connected networks and modern convolutional architectures.

| Feature | Custom MLP (C++) | CNN (TensorFlow/ONNX) |
| :--- | :--- | :--- |
| **Architecture** | Fully Connected (Dense Layers) | Convolutional + Pooling Layers |
| **Input Handling** | Flattened Vector (1D array of pixels) | Matrix / Tensor (Preserves 2D structure) |
| **Spatial Awareness** | ❌ **None.** Pixels are treated independently. | ✅ **High.** Recognizes shapes, edges, and textures locally. |
| **Translation Invariance** | Low. Moving the hand slightly changes the input vector completely. | High. Recognizes features regardless of position. |
| **Complexity** | Simple math, implemented from scratch to understand gradients. | Complex, requires libraries (TensorFlow/OpenCV). |
| **Accuracy** | Good for centered, static images. Struggles with variations. | Excellent. Robust against noise and position shifts. |

---

## 🧠 Theoretical Background

### 1. Multilayer Perceptron (Custom C++ Implementation)

The custom neural network is a "Feedforward Neural Network" built using matrix operations via the **Eigen3** library. It processes the input image ($32 \times 32$ pixels) as a flattened vector of 1024 distinct values.

#### Architecture Topology
* **Input Layer:** 1024 neurons.
* **Hidden Layer 1:** 256 neurons.
* **Hidden Layer 2:** 64 neurons.
* **Output Layer:** 8 neurons (representing the 8 gesture classes).

#### Forward Propagation
Each neuron performs a weighted sum of its inputs ($Z$) and applies a non-linear activation function ($\sigma$). For a single layer $l$:

$$
Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}
$$

$$
A^{[l]} = \sigma(Z^{[l]})
$$

**Implementation in C++ (`NeuralNet.h`):**
```cpp
// Matrix multiplication and addition of bias
VectorXd h1 = (W1 * input + b1);

// Apply Activation Function (Sigmoid)
h1 = h1.unaryExpr([&](double x){ return sigmoid(x); });
```

#### Activation Function: Sigmoid
This project uses the Sigmoid function to squash values between 0 and 1.

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The derivative of the sigmoid function, which is crucial for the learning process (Backpropagation), has a convenient mathematical property:

$$
\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))
$$

```cpp
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}
```

#### Learning: Backpropagation
The network learns by minimizing a Cost Function (Mean Squared Error). We calculate the gradient of the error with respect to the weights using the **Chain Rule** and update the weights (Gradient Descent).

$$
\frac{\partial C}{\partial w} = \underbrace{\frac{\partial C}{\partial a}}_{\text{Error from next layer}} \cdot \underbrace{\frac{\partial a}{\partial z}}_{\text{Activation derivative}} \cdot \underbrace{\frac{\partial z}{\partial w}}_{\text{Input from prev. layer}}
$$

**Implementation in C++ (`NeuralNet.h`):**
```cpp
// Calculate error at output
VectorXd outputError = target - out;

// Calculate gradient (Error * Derivative of Sigmoid)
VectorXd outputGradient = outputError.array() * out.unaryExpr([&](double x){ return sigmoidDerivative(x); }).array();

// Update Weights (Learning Rate * Gradient * Input Transposed)
W3 += learningRate * outputGradient * h2.transpose();
```

---

### 2. Convolutional Neural Network (CNN)

While the MLP treats the image as an unstructured list of numbers, the CNN preserves the spatial structure (height, width, channels).

#### The Convolution Operation
Instead of fully connected weights, the CNN uses learnable **filters (kernels)**. A kernel slides over the input image to produce feature maps.

$$
(I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m, n)
$$

#### Network Definition (Python/Keras)
The architecture consists of three convolutional blocks followed by a dense classifier.

```python
model = models.Sequential([
    # Feature Extraction
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Classification
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax') // Softmax for probability distribution
])
```

#### Deployment via ONNX
The trained model is exported to the **Open Neural Network Exchange (ONNX)** format, allowing it to be loaded in C++ using OpenCV's DNN module without requiring TensorFlow at runtime.

```cpp
// Loading the model in C++
cv::dnn::Net net = cv::dnn::readNetFromONNX("hand_cnn.onnx");

// Preparing the input blob (1x32x32x1)
cv::Mat blob = cv::dnn::blobFromImage(grayInput, 1.0/255.0, cv::Size(32, 32), ...);
net.setInput(blob);
```

---

## 🚀 How to Use (Workflow)

To reproduce the results or train with your own data, follow this strict order:

### Step 1: Data Collection 📸
* Navigate to `01_Data_Collector`.
* Compile and run the C++ program.
* **Controls:**
    * **'B'**: Capture background (ensure no hand is in the frame).
    * **'0'-'7'**: Select the class label (Fist, Peace, etc.).
    * **SPACE (Hold)**: Record images. They are saved into labeled subfolders automatically.

### Step 2: Augmentation & Training 🐍
* Navigate to `02_Data_Augmenter`.
* Run `augment_data.py`: This script reads the raw images, applies rotations and noise, and saves them to `training_data_final`.
* Run `train_cnn.py`: This uses TensorFlow/Keras to train the CNN on the augmented data.
* **Result:** A file named `hand_cnn.onnx` will be generated.

### Step 3: Deployment (Inference) ⚡

**Option A: The CNN (Recommended)**
* Navigate to `04_Model_CNN_Inference`.
* Copy the `hand_cnn.onnx` file into the build directory (next to the executable) AND the project directory.
* Run the program. It uses OpenCV's DNN module to load the ONNX file for robust prediction.

**Option B: The Custom NN**
* Navigate to `03_Model_CustomNN`.
* This module reads the augmented data directly from the folders and performs training in C++ (Backpropagation) before switching to live inference mode.

---

## 🛠 Dependencies

**C++ Projects:**
* **OpenCV 4.x** (Core, HighGUI, ImgProc, DNN)
* **Eigen3** (Only for the Custom NN project)
* C++20 Compiler (MSVC, GCC, or Clang)

**Python Scripts:**
* Python 3.x
* `tensorflow` (with `tf_keras`)
* `numpy < 2.0`
* `opencv-python`
* `tf2onnx`

## 📝 License
This project is open-source. Feel free to use and modify.
