# cs448_final
---

Dataset: https://www.kaggle.com/code/paoloripamonti/twitter-sentiment-analysis/input


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


# Label encoding for target
label_encoder = LabelEncoder() 
df['target'] = label_encoder.fit_transform(df['target'])


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