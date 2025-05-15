# CNN-Exercises

1. Project Overview
This project uses PyTorch to build and train a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. It can recognize 10 classes including cats, dogs, airplanes, cars, and more.  

2. Model Architecture
The model consists of 4 convolutional layers with BatchNorm, ReLU, and MaxPooling, followed by 4 fully connected layers with Dropout to prevent overfitting. It uses AdaptiveAvgPool2d for global feature aggregation and outputs probabilities for 10 classes.  

3. Training Configuration
Optimizer: AdamW
Learning rate: 0.001
Loss function: CrossEntropyLoss
Epochs: 40
Device: Auto (GPU if available)
Training progress is logged using TensorBoard.

4. result
The correct result is 74%. This is an interesting practice to understand the CNN model and you can fix the hyperparameters which make it more robust.
