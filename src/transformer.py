import torch
from torch import nn

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

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=512,
            dropout=0.1
        )

        # Linear layer for classification
        self.fc = nn.Linear(embed_size, num_classes)

        self.context_size = context_size

    def forward(self, x):
        # Assuming x is the input tensor of shape (sequence_length, batch_size)
        # AND x must be 230 tokens long

        # Embedding layer
        x = self.embedding(x)

        # Transformer layer
        x = self.transformer(x, x)

        # Global average pooling
        x = x.mean(dim=0)

        # Classification layer
        x = self.fc(x)

        return x
