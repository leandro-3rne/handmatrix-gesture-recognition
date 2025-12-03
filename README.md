# HandMatrix: Gesture Recognition Engine ✋🤖

This project is a real-time hand gesture recognition system capable of identifying 8 different hand signs (like Fist, Peace, Rock, etc.) via webcam. It serves as an educational deep dive into Computer Vision and Machine Learning, contrasting a manually implemented neural network (MLP) with a state-of-the-art CNN approach. Essentially, it translates raw pixel data from your camera into actionable commands or classifications instantly.

<p align="center">
  <img src="HandMatrix/README_Images/Fist_Gesture.png" width="20%" title="Fist Gesture">
  <img src="HandMatrix/README_Images/Peace_Gesture.png" width="20%" title="Peace Gesture">
  <img src="HandMatrix/README_Images/Hand_Gesture.png" width="20%" title="Hand Gesture">
  <img src="HandMatrix/README_Images/Up_Gesture.png" width="20%" title="Up Gesture">
  <img src="HandMatrix/README_Images/Down_Gesture.png" width="20%" title="Down Gesture">
  <img src="HandMatrix/README_Images/Rock_Gesture.png" width="20%" title="Rock Gesture">
  <img src="HandMatrix/README_Images/Chill_Gesture.png" width="20%" title="Chill Gesture">
  <img src="HandMatrix/README_Images/Middle_Gesture.png" width="20%" title="Middle Gesture">
</p>

![C++](https://img.shields.io/badge/C++-20-blue.svg) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg) ![Python](https://img.shields.io/badge/Python-3.12-yellow.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)

## 📂 Project Overview

This repository is structured into four distinct modules:

1.  **Data Collector (C++):** A specialized tool to capture, pre-process, and label grayscale hand gestures into organized subfolders.
2.  **Data Augmentation (Python):** A script to augment raw data (rotation, noise) to create a robust dataset.
3.  **Model NN (C++):** A Multilayer Perceptron implemented purely in C++ using the Eigen3 library, featuring manual Backpropagation.
4.  **Model CNN (Python & C++):** Contains the Python training script (TensorFlow) and the C++ inference engine (OpenCV DNN) using the exported ONNX model.

---

## 👁️ The Computer Vision Pipeline

Before any Neural Network can classify a gesture, the raw webcam image must be heavily processed to extract relevant features and reduce noise. Both models use the exact same preprocessing pipeline to ensure consistency. The OpenCV Library was used here, to ensure a standardized and efficient pipeline for capturing, processing, and normalizing the webcam feed for the neural networks.

### 1. Region of Interest (ROI) & Background Subtraction
We focus only on a specific area (`scanBox`) to reduce computational load. To isolate the hand from the background, we calculate the absolute difference between the current frame and a stored background frame.

```cpp
// Calculate difference between current frame (cleanRoi) and stored background
cv::absdiff(cleanRoi, background, diff);

// Create a binary mask: Pixels with high difference become white (255), others black (0)
cv::threshold(diff, mask, 30, 255, cv::THRESH_BINARY);
```

### 2. Feature Extraction (Double Canny Trick)
Standard Canny edge detection often only captures the outline of the hand. To get more internal details (like wrinkles or finger separation) and fill the shape better, I experimented with a sequence: **Canny -> GaussianBlur -> Canny**. This "double Canny" approach tends to pick up more texture and creates a denser representation of the hand, almost "filling" it with features rather than just outlining it.

```cpp
// Convert to HSV and extract Saturation channel (channel 1)
cv::cvtColor(processingImg, hsv, cv::COLOR_BGR2HSV);
cv::split(hsv, channels);
cv::Mat saturation = channels[1];

// First Pass
cv::Canny(saturation, edges, 50, 150);
// Blur to merge close edges
cv::GaussianBlur(edges, edges, cv::Size(5, 5), 0);
// Second Pass to re-sharpen
cv::Canny(edges, edges, 50, 150);
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

![Basic architecture of a simple NN](HandMatrix/README_Images/NN_Image.png)

#### Architecture Topology
* **Input Layer:** 1024 neurons ($32 \times 32$ pixels flattened).
* **Hidden Layer 1:** 256 neurons.
* **Hidden Layer 2:** 64 neurons.
* **Output Layer:** 8 neurons (representing the 8 gesture classes).

**Note on Dimensions:**
The network currently uses "depth" (multiple layers) to learn hierarchical features. You could also increase the "width" (e.g., 512 or 1024 neurons per hidden layer). Wider layers can memorize more patterns but are prone to overfitting and require more computation. Deeper networks (more layers) generally learn more complex, abstract abstractions but are harder to train (vanishing gradient problem).

**Implementation in C++ (`NeuralNet.h`):**
```cpp
// Definitions in NeuralNet.h using Eigen3
Eigen::MatrixXd W1, W2, W3; // Weights
Eigen::VectorXd b1, b2, b3; // Biases

// Constructor Initialization
// Matrix Dimensions: [Rows = Neurons in current layer, Cols = Neurons in previous layer]
NeuralNet(int inputSize, int hidden1Size, int hidden2Size, int outputSize) {
    
    // Layer 1: Input (1024) -> Hidden 1 (256)
    W1 = Eigen::MatrixXd::Random(hidden1Size, inputSize);  
    b1 = Eigen::VectorXd::Random(hidden1Size);             

    // Layer 2: Hidden 1 (256) -> Hidden 2 (64)
    W2 = Eigen::MatrixXd::Random(hidden2Size, hidden1Size); 
    b2 = Eigen::VectorXd::Random(hidden2Size);              

    // Layer 3: Hidden 2 (64) -> Output (8)
    W3 = Eigen::MatrixXd::Random(outputSize, hidden2Size);  
    b3 = Eigen::VectorXd::Random(outputSize);               
}
```

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

#### Activation Functions
Activation functions introduce non-linearity, allowing the network to learn complex data.
* **Sigmoid (Used in my MLP):** Maps values to (0, 1). Smooth gradient, good for probability-like outputs. *Cons:* Can lead to vanishing gradients in deep networks.

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The derivative of the sigmoid function, which is crucial for the learning process (Backpropagation), has a convenient mathematical property:

$$
\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))
$$


#### Learning: Backpropagation
The network learns by minimizing a Cost Function (Mean Squared Error). We use **Gradient Descent** to update weights. The gradient is calculated using the **Chain Rule**. To find how much a specific weight $w$ contributes to the error $C$, we calculate:

$$
\frac{\partial C}{\partial w} = \underbrace{\frac{\partial C}{\partial a}}_{\text{Error from next layer}} \cdot \underbrace{\frac{\partial a}{\partial z}}_{\text{Activation derivative}} \cdot \underbrace{\frac{\partial z}{\partial w}}_{\text{Input from prev. layer}}
$$

**For Multiple Layers:**
The error is propagated backward layer by layer. The error for a hidden layer is calculated based on the error of the *following* layer, weighted by the connections between them.

$$
\delta^{[l]} = (W^{[l+1]})^T \cdot \delta^{[l+1]} \odot \sigma'(z^{[l]})
$$

Where $\delta$ is the error term for a layer.

**The Weight Update Step:**
Once the gradients are calculated, we update the weights to minimize the error. We subtract a portion of the gradient from the current weights:

$$
W_{new} = W_{old} - \eta \cdot \frac{\partial C}{\partial W}
$$

* **$\frac{\partial C}{\partial W}$ (Gradient):** Indicates the direction to increase the error. We subtract it to go the opposite way (downhill).
* **$\eta$ (Learning Rate):** A hyperparameter that controls the **size of the step** we take.
    * *If too high:* The network might overshoot the minimum and fail to learn.
    * *If too low:* Training becomes very slow as the network takes tiny steps.
      

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

![Basic architecture of a simple CNN](HandMatrix/README_Images/CNN_Image.png)

#### The Convolution Operation (Kernels)
Instead of fully connected weights, the CNN uses learnable **filters (kernels)**. A kernel "slides" over the input image pixel by pixel. At each step, it performs element-wise multiplication with the image patch and sums the results.

**What does a Kernel look like?**
A kernel is simply a small matrix of weights. For example, this $3 \times 3$ kernel detects vertical edges (sobel filter):

$$
K = \begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}
$$

When this matrix slides over an image, the dot product will be high where there is a vertical edge (difference between left and right pixels) and zero where the area is flat. The CNN learns these numbers automatically during training!

$$
(I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m, n)
$$

#### Pooling (Downsampling)
To simplify the information and make the model robust to small shifts, we use **Max Pooling**. Imagine a $2 \times 2$ window sliding over the feature map. It only keeps the *largest* value in that window and discards the rest. This reduces the image size by half (e.g., from $32 \times 32$ to $16 \times 16$) while keeping the most important feature (e.g., "there is a strong edge here").

#### Network Definition (Python/Keras)
The architecture consists of three convolutional blocks followed by a dense classifier.

**1. The Convolutional Blocks (The "Eyes")**
These blocks act as **feature extractors**. They scan over the image to "see" patterns.
* **Block 1:** Detects simple features like **edges, lines, and corners**.
* **Block 2 & 3:** Combine those lines to recognize complex shapes like **curves, fingers, or hand outlines**.

**2. The Dense Classifier (The "Brain")**
This part acts as the **decision maker**. (Similar to the MLP)
* **Flattening:** It takes the square feature maps from the convolutional blocks and stretches them into a long list of numbers (a vector).
* **Dense Layers:** It analyzes this list to decide: *"Based on these curves and lines, there is a 99% chance this is a 'Peace' sign."*

**Implementation in Python (`train_cnn.py`):**
```python
model = models.Sequential([
    # Feature Extraction
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Classification
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax') // Softmax for probability distribution
])
```

#### Activation Function: ReLU & Softmax
* **ReLU:** Used in hidden layers to introduce non-linearity without the vanishing gradient problem.
* **Softmax:** Used in the final layer to output probabilities (e.g., Fist: 0.1, Peace: 0.8, ...) that sum up to 1.

$$
f(x) = \max(0, x) \quad \text{(ReLU)}
$$

$$
P(y=j \mid x) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}} \quad \text{(Softmax)}
$$

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

### Step 2: Data Augmentation 🐍
* Navigate to `02_Data_Augmenter`.
* Run `augment_data.py`.
* This script reads the raw images from the collector, applies rotations and noise, and saves the result to a new folder (`training_data_final`).

### Step 3: CNN Training (Python) 🧠
* Navigate to `04_Model_CNN`.
* Run `train_cnn.py`.
* This script loads the augmented data, trains the TensorFlow model, and exports it as `hand_cnn.onnx` directly into this folder.

### Step 4: Inference / Live Demo ⚡

**Option A: The CNN (Recommended)**
* Navigate to `04_Model_CNN`.
* Compile the C++ project.
* Ensure `hand_cnn.onnx` is in the same directory as the executable.
* Run the program to see the CNN predicting gestures in real-time.

**Option B: The Custom NN**
* Navigate to `03_Model_NN`.
* This module reads the augmented data directly from the folders and performs training in C++ (Backpropagation) before switching to live inference mode.

## 📂 Datasets

If you want to skip data collection and jump straight to training, you can use my pre-recorded datasets.

* **`00_Datasets/raw_data.zip`**: Contains the original grayscale images captured with the C++ Data Collector.
* **`00_Datasets/augmented_data.zip`**: Contains the fully processed, rotated, and noisy images ready for CNN training.

(Make sure to unzip the files first!)

---

## 🛠 Dependencies

**C++ Projects:**
* **OpenCV 4.x** (Core, HighGUI, ImgProc, DNN)
* **Eigen3** (Only for the Custom NN project)
* C++20 Compiler (MSVC, GCC, or Clang)

**Python Scripts:**
* **Python 3.9 - 3.12** (Python 3.13 is not yet supported by TensorFlow!)
* `tensorflow` (with `tf_keras`)
* `numpy < 2.0`
* `opencv-python`
* `tf2onnx`
* `protobuf` (Version 4.x recommended)

To install Python dependencies:
```bash
pip install -r 02_Data_Augmenter/requirements.txt
```

## 📝 License
This project is open-source. Feel free to use and modify.
