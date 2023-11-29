# cs448_final
---

Dataset: https://www.kaggle.com/code/paoloripamonti/twitter-sentiment-analysis/input

   ## Sentiment Analysis using Simple RNN

### Overview
This repository contains a sentiment analysis model implemented in PyTorch, utilizing a simple recurrent neural network (RNN). The model is trained on a labeled dataset of tweets, aiming to predict sentiment polarity. Key components include data preprocessing, model architecture, training, and evaluation loops.

### Dataset
The dataset, sourced from "./data/data.csv," undergoes preprocessing using the TorchText library. A subset of the data is loaded and tokenized, and a vocabulary is built. Challenges arise from label inaccuracies in the dataset, as some tweets with neutral sentiment are labeled as positive or negative.

# RNN Text Classification with PyTorch

This project demonstrates a simple implementation of a Recurrent Neural Network (RNN) for text classification using PyTorch. The goal is to classify text data into one of two classes (pos or neg sentiment).

## Overview

The implementation consists of the following components:

1. Data Processing: Tokenizing text data, mapping tokens to indices, and padding sequences.
2. Model Architecture: A basic RNN model designed for text classification.
3. Training: Using cross-entropy loss and Adam optimizer to train the model.
4. Evaluation: Calculating accuracy on a separate test set.

## Data Processing

### Data Collection and Label Encoding

The data is loaded from a CSV file using the `pandas` library. The target labels are encoded using `LabelEncoder`.

### Model Architecture
The sentiment analysis model is built with a simple RNN, comprising an embedding layer, an RNN layer, and a linear layer. The RNN processes packed sequences, allowing for efficient handling of variable-length input.

### Training and Evaluation
The model is trained using stochastic gradient descent with a binary cross-entropy loss. Training progress is visualized with tqdm progress bars. The evaluation loop calculates accuracy on a separate validation set, revealing the model's ability to generalize.

# Label encoding for target
```python
label_encoder = LabelEncoder() 
df['target'] = label_encoder.fit_transform(df['target'])
```

### Tokenization and Vocabulary Building

The `get_data_torchtext` function in the utils file processes the text data. It uses the TorchText library for tokenization and building a vocabulary.

```python
# Torchtext tokenizer
tokenizer = get_tokenizer("basic_english")

# Define a generator function to yield tokens
def yield_tokens(data):
    for _, text in data:
        yield tokenizer(text)

# Build vocabulary
vocab = build_vocab_from_iterator(yield_tokens(data), specials=vocab_specials)
vocab.set_default_index(vocab[vocab_specials[0]])
```

## Model Architecture

The RNN model is defined by the `SimpleRNNModel` class. It includes:
- An embedding layer to convert token indices to dense vectors.
- A one-layer RNN to process the embedded sequences.
- A fully connected layer for classification using the output from the last time step.

## Training

The model is trained using the cross-entropy loss function and the Adam optimizer. Training is performed in a loop using a DataLoader with batched data.

## Evaluation

The trained model is evaluated on a separate test set, and accuracy is calculated.

## Usage

1. Install the required dependencies:

   ```bash
   pip install torch tqdm scikit-learn
   ```

### Challenges and Solutions
To address padding-related runtime errors, the code implements a more stable padding approach. Sorting the batch by descending lengths resolves dimension size issues, ensuring smooth execution.

### Results
After five epochs, the model achieves a validation accuracy of 76.62%. This promising result is presented along with ongoing work on visualizations, including training vs. validation loss plots and example predictions.

### Conclusions
Despite dataset labeling challenges, the model demonstrates robustness and potential for improvement. Observations about dataset limitations are insightful, providing context for the achieved accuracy. This contribution significantly enhances the model's usability and sets the stage for further optimizations.