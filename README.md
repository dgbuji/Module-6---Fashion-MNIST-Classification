# Module-6---Fashion-MNIST-Classification
Module 6 Assignment: Fashion MNIST Classification

# Fashion MNIST Image Classification with CNN

## Assignment Overview

As a junior machine learning researcher at Microsoft AI, this project focuses on classifying images from the Fashion MNIST dataset using a Convolutional Neural Network (CNN). This serves as a foundational step for future work involving user profile image classification for targeted marketing.

## Assignment Description

This submission addresses the following tasks:

1.  **Convolutional Neural Network (CNN) Development:** A six-layer CNN was developed using Keras in Python to classify the Fashion MNIST dataset.
2.  **Prediction:** Predictions are made for at least two images from the Fashion MNIST test dataset, demonstrating the model's classification capability.

* `fashion_mnist_cnn.ipynb`: Contains all the Python code for CNN implementation, training, evaluation, and prediction.

## Prerequisites for Python Implementation

To run this project, you need Python installed (version 3.7 or higher recommended) along with the following libraries:

* `tensorflow` (includes Keras)
* `numpy`
* `matplotlib`

You can install these dependencies using pip:

```bash
pip install tensorflow numpy matplotlib

## Prerequisites for R Implementation

Prerequisites
R (version 4.0 or newer recommended)
RStudio (recommended IDE)
An active internet connection for package downloads.
Environment Setup
The R keras package relies on a Python installation of TensorFlow. This setup process can sometimes be tricky but is essential.

Install the R keras package:
Open your R console/RStudio and run:

R

install.packages("keras")
Watch for any error messages during installation. If there are, resolve them before proceeding (e.g., internet connection, R version compatibility).
Install TensorFlow and its Python dependencies:
This is the most crucial step. It will set up the necessary Python environment.

Restart your R session in RStudio (Session -> Restart R, or Ctrl+Shift+F10 / Cmd+Shift+F10). This is very important to avoid DLL conflicts.
After restarting, immediately run these two lines:
R

library(keras)       # Load the keras package into your session
install_keras()      # This installs TensorFlow and other Python dependencies
Follow the prompts: install_keras() will ask you to confirm installations and might offer to create a Python virtual environment (e.g., r-tensorflow). It's generally recommended to accept the defaults. This process can take several minutes depending on your internet speed and system.
Troubleshooting install_keras():
If you get Error in install_keras(): could not find function "install_keras": This means library(keras) was not run or failed. Ensure library(keras) runs without error before install_keras().
If you get 'r-tensorflow' exists but is not a virtual environment or similar errors:
Restart your R session.
Manually navigate to the directory mentioned in the error (e.g., C:\Users\YourUser\OneDrive - Nexford University\Documents\.virtualenvs\).
Delete the r-tensorflow folder if it exists.
Restart R again, then run library(keras) followed by install_keras() (or install_keras(envname = "my_new_env") to use a different name).
Code Structure and Steps
The R code is structured into sequential steps, mirroring the typical machine learning workflow.

Step 1: Setup and Library Loading
Loads the necessary R libraries, primarily keras and tensorflow.
Sets up the random seed for reproducibility.
Step 2: Load the Dataset
Uses dataset_fashion_mnist() from keras to download and load the Fashion MNIST training and test datasets.
Prints the initial dimensions of the image and label arrays.
Includes code to display a sample image from the training set, demonstrating raw image loading.
Step 3: Data Pre-processing
Normalization: Scales image pixel values from (0-255) to (0-1). This helps neural networks learn more effectively.
Reshaping for CNN: Reshapes the image arrays from (samples, 28, 28) to (samples, 28, 28, 1). The 1 represents the single grayscale channel, which is required by Keras CNN layers.
One-Hot Encoding: Converts numerical labels (0-9) into a one-hot encoded format (e.g., 2 becomes [0,0,1,0,0,0,0,0,0,0]).
Includes code to display pre-processed images, ensuring they are properly formatted.
Step 4: Build the Model
Defines the architecture of the Convolutional Neural Network using keras_model_sequential().
Consists of:
Convolutional Layers (layer_conv_2d) with ReLU activation for feature extraction.
Max Pooling Layers (layer_max_pooling_2d) for downsampling.
Flatten Layer (layer_flatten) to convert 2D feature maps into a 1D vector.
Dense (Fully Connected) Layers (layer_dense) for classification, with the final layer using softmax activation for multi-class probabilities.
Compiles the model with an optimizer (adam), loss function (sparse_categorical_crossentropy if labels are not one-hot encoded, or categorical_crossentropy if they are), and evaluation metric (accuracy).
Step 5: Train the Model
Trains the compiled model using the fit() method.
Specifies training data, epochs (number of passes through the entire dataset), and validation data (a portion of the training data used to monitor performance during training).
Step 6: Evaluate the Model
Evaluates the trained model's performance on the unseen test_images and test_labels.
Prints the test loss and test accuracy, providing an unbiased estimate of the model's generalization ability.
Step 7: Make Predictions
Uses the trained model to make predictions on a few sample images from the test set.
Displays the image along with its true label and the model's predicted label.
Includes specific R array manipulation (t(apply(..., 2, rev))) to correctly orient the images for plotting in R.


