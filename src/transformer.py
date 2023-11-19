import math

import torch
from torch.nn import functional as F
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, context_size, device):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, context_size).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.positional_encoding = torch.zeros((context_size, d_model)).to(device)
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.positional_encoding = self.positional_encoding.unsqueeze(1)

    def forward(self, x):
        return x + self.positional_encoding[:x.size(0), :].detach()

class TransformerModel(nn.Module):
    def __init__(
            self,
            vocab_size,  # Count of unique tokens (for embedding layer)
            embed_size,  # What dimensionality do we want the embedding to have?
            num_heads,   # Number of attention heads
            num_layers,  # Number of encoding / decoding layers
            context_size,# Size of input (230 because our max tweet size is 229)
            num_classes
        ):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.pos_enc = PositionalEncoding(embed_size, context_size, torch.device("cuda"))

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=512,
            dropout=0.1
        )

        # Linear layer for classification
        self.l1 = nn.Linear(embed_size, 64)
        self.l2 = nn.Linear(64, num_classes)

        self.context_size = context_size

    def forward(self, x):
        # Assuming x is the input tensor of shape (sequence_length, batch_size)

        # Embedding layer
        x = self.embedding(x)
        
        x = self.pos_enc(x)
        
        # Transformer layer
        x = self.transformer(x, x)
        
        # Global average pooling
        x = x.mean(dim=0)
        
        # Classification layer
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)

        return F.softmax(x, dim=1)

class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(SimpleTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = x.t()
        # x shape: (sequence_length, batch_size)

        # Embedding layer
        x = self.embedding(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Apply softmax activation for classification
        x = F.softmax(x, dim=1)

        return x
