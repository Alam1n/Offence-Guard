# Offence-Guard
Offence guard is a machine learning project that is able to detect weather a statement is offensive or not.


This repository contains the code and files for a binary text classification model designed to detect offensive and non-offensive language.

### Key Features

* Model:
* Accuracy: The model achieves approximately 94% accuracy in classifying two classes
* Loss: The model achieves approximate loss of 0.15.

### Model Architecture
The model uses a simple neural network architecture consisting of the following layers:

* Embedding Layer: Converts words into dense vectors of fixed size (128-dimensional embeddings).
* Global Average Pooling Layer: Reduces the dimensionality by averaging across the sequence, summarizing the entire text into a single vector.
* Dense Layer: A fully connected layer with 64 neurons and ReLU activation.
* Output Layer: A Dense layer with 2 neurons and softmax activation for binary classification (offensive vs. non-offensive).
* This architecture efficiently handles text data and leverages Global Average Pooling to capture semantic information from the embedded input.
