import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from utils import get_data_torchtext

# Loading data and preprocessing
file_path = "./data/data.csv"
vocab_specials = ["<unk>"]
vocab, tokenizer, data = get_data_torchtext(file_path=file_path, vocab_specials=vocab_specials)

# Defining the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_embedded)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Applying the linear layer to the output of the packed sequence
        hidden = hidden.squeeze(0)
        hidden = hidden[lengths - 1, range(len(lengths))] # getting the last output for each sequence
        return self.fc(hidden)

# Splitting the data into training and validation sets
split_ratio = 0.8
split_index = int(len(data) * split_ratio)
train_data, valid_data = data[:split_index], data[split_index:]

# Initializing the model, optimizer, and loss function
input_dim = len(vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 1

model = SimpleRNN(input_dim, embedding_dim, hidden_dim, output_dim)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

# Function to numericalize the text
def numericalize_text(text):
    return [vocab[token] for token in tokenizer(text)]

# Using DataLoader for efficient batch handling
collate_fn = lambda batch: (
    torch.tensor([item[0] for item in batch], dtype=torch.float),  # labels
    pad_sequence([torch.tensor(numericalize_text(item[1])) for item in batch], batch_first=True),  # text
    [len(item[1]) for item in batch]  # lengths
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_data, batch_size=64, shuffle=False, collate_fn=collate_fn)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    for labels, text, lengths in train_loader:
        optimizer.zero_grad()
        predictions = model(text, lengths).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

# Evaluation loop
model.eval()
correct_predictions = 0
total_examples = 0

with torch.no_grad():
    for labels, text, lengths in valid_loader:
        predictions = model(text, lengths).squeeze(1)
        rounded_predictions = torch.round(torch.sigmoid(predictions))
        correct_predictions += (rounded_predictions == labels).sum().item()
        total_examples += labels.size(0)

accuracy = correct_predictions / total_examples
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
