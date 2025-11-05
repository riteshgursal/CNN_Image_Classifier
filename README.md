# Image Classification using Convolutional Neural Networks (CNN)

This project is a mini-project for classifying images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN). The model is built with Python, TensorFlow, and Keras.

This project demonstrates a foundational understanding of building, training, and evaluating a deep learning model for computer vision.

**Project Status:** Completed
**Accuracy Achieved:** ~85% (with 30 epochs)
                       ~69.27% (with 10 epochs)

## 🚀 Tech Stack

* **Python:** The core programming language.
* **TensorFlow:** The primary deep learning library.
* **Keras:** The high-level API used to build the model.
* **Matplotlib:** Used for performance visualization (plotting accuracy and loss).
* **NumPy:** For numerical operations and data handling.

## 📊 Dataset: CIFAR-10

The model is trained on the **CIFAR-10 dataset**, a classic benchmark in machine learning.
* It consists of **60,000** 32x32 color images.
* There are **10 classes**: 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'.
* **50,000** images are used for training and **10,000** for testing.

## 🛠️ Model & Techniques

This project implements several key techniques to build a robust classifier.

### 1. CNN Architecture
The model is a sequential stack of layers:
* **Data Augmentation:** `RandomFlip`, `RandomRotation`, and `RandomZoom` layers are used to create more robust and varied training data. This is a key technique to prevent overfitting.
* **Convolutional Base:** A series of `Conv2D` (feature-finding) and `MaxPooling2D` (down-sampling) layers to learn spatial hierarchies of features from the images.
* **Classifier Head:**
    * A `Flatten` layer to convert the 2D feature maps into a 1D vector.
    * A `Dense` layer (128 units, `relu` activation) to learn high-level patterns.
    * A `Dropout` layer (0.5) to further prevent overfitting by randomly "turning off" neurons during training.
    * The final `Dense` output layer (10 units, `softmax` activation) to produce a probability distribution for each of the 10 classes.

### 2. Performance Visualization
The script generates plots for **Model Accuracy** and **Model Loss** over each epoch. This is crucial for:
* Visualizing how the model learns.
* Comparing training and validation performance.
* Diagnosing problems like **overfitting** (when validation loss starts to increase while training loss decreases).


## 📈 Results

After 30 epochs of training, the model achieves a test accuracy of **~85%**.

*(Note: Running with fewer epochs, like 10, will result in lower accuracy but provides a faster test of the code.)*
