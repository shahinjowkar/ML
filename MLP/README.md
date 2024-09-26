# Multi-Layer Perceptron (MLP) for Sign Language MNIST Classification

This repository contains the implementation of a **Multi-Layer Perceptron (MLP)** from scratch to classify images from the **Sign Language MNIST** dataset. The model utilizes various activation functions, including ReLU, Leaky ReLU, and Sigmoid, and is trained using **mini-batch gradient descent**. The project also explores the impact of network depth, activation functions, and regularization on model performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Implementation Details](#implementation-details)
- [Activation Functions](#activation-functions)
- [Optimization](#optimization)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Project Overview
This project implements an MLP from scratch, focusing on classifying hand gestures from the **Sign Language MNIST** dataset. The model is trained with **mini-batch gradient descent**, and different configurations of the network are tested, including models with different numbers of hidden layers and activation functions. The performance of the MLP is compared to a convolutional neural network (CNN) built using existing libraries like TensorFlow.

## Dataset
The dataset used is the **[Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist/data)**, which consists of 28x28 grayscale images of hand signs corresponding to letters of the alphabet.

- **Preprocessing**: 
  - The images are vectorized, normalized, and split into training and testing sets.

## Implementation Details
The MLP is implemented from scratch without the use of machine learning libraries such as TensorFlow or PyTorch. The following features are included:

- **Input Layer**: Vectorized 28x28 pixel images (784 input units).
- **Hidden Layers**: Experiments are conducted with 1 and 2 hidden layers, with {32, 64, 128, 256} hidden units.
- **Output Layer**: 26 output units corresponding to the 26 letters in the Sign Language alphabet.

### 1. Forward Pass
- Linear transformation is applied at each layer: 
  \[
  z^{[l]} = W^{[l]} \cdot a^{[l-1]} + b^{[l]}
  \]
  Where:
  - \( z^{[l]} \) is the linear activation of layer \( l \)
  - \( W^{[l]} \) are the weights, and \( b^{[l]} \) are the biases for layer \( l \)

### 2. Activation Functions
Three types of activation functions were used and compared:
- **ReLU** (Rectified Linear Unit): 
  \[
  f(z) = \max(0, z)
  \]
  - Used for most hidden layers due to its effectiveness in deep learning.
  
- **Leaky ReLU**: 
  \[
  f(z) = \max(0.01z, z)
  \]
  - Used to address the "dying ReLU" problem by allowing a small, non-zero gradient when \( z \) is negative.

- **Sigmoid**: 
  \[
  f(z) = \frac{1}{1 + e^{-z}}
  \]
  - Traditionally used for binary classification, included in this project for experimentation in multi-class settings.

### 3. Backpropagation
- **Gradient Descent**: Backpropagation is performed using **mini-batch gradient descent** to update the weights.
  \[
  W^{[l]} = W^{[l]} - \alpha \cdot \frac{\partial J}{\partial W^{[l]}}
  \]
  - Where \( \alpha \) is the learning rate and \( \frac{\partial J}{\partial W^{[l]}} \) is the gradient of the loss function with respect to the weights.

### 4. Regularization
- **L2 Regularization**: Applied to prevent overfitting by penalizing large weights.
  \[
  \lambda \sum_{l=1}^{L} ||W^{[l]}||_2^2
  \]
  - Where \( \lambda \) is the regularization parameter.

## Optimization
- **Mini-batch Gradient Descent**: The training process is optimized using mini-batches, which helps improve convergence speed and reduce memory usage.
- **Hyperparameter Tuning**: Various hyperparameters such as learning rate, batch size, number of hidden layers, and number of units in each layer are tuned for optimal performance.

## Evaluation Metrics
The performance of the models is evaluated using the following metrics:

- **Accuracy**: The proportion of correctly classified images.
- **Loss**: Cross-entropy loss is used as the loss function:
  \[
  J(W) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
  \]
- **Comparison with CNN**: As part of the experiments, a CNN built using TensorFlow is trained on the same dataset and compared with the MLP in terms of accuracy and speed.

## Usage
To run the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/shahinjowkar/ML.git
    ```
2. Navigate to the project directory and open the Jupyter notebook:
    ```bash
    jupyter notebook MLP.ipynb
    ```
3. Ensure the required libraries (e.g., NumPy, Pandas, Matplotlib) are installed:
    ```bash
    pip install numpy pandas matplotlib
    ```
4. Run the notebook to train the models and evaluate their performance.

## Results
The results show that:
- **MLP with ReLU activation** outperforms Sigmoid and Leaky ReLU in terms of test accuracy.
- **Mini-batch gradient descent** provides faster convergence compared to full-batch gradient descent.
- The CNN achieves better accuracy than the MLP due to its ability to capture spatial relationships in the image data.

Plots for training and test accuracy, as well as loss curves, are available in the notebook.

## References
- **Sign Language MNIST Dataset**: [Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist/data)
- **Neural Network Basics**: [CS231n](https://cs231n.github.io/neural-networks-2/)
- **Scipy Gradient Check**: [Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.check_grad.html)
