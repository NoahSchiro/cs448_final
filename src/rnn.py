import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from utils import get_data_rnn 
from tqdm import tqdm

# Loading data and preprocessing
file_path = "./data/data.csv"
vocab_specials = ["<unk>"]
vocab, tokenizer, data = get_data_rnn(file_path=file_path, vocab_specials=vocab_specials)

# Defining the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=True)
        packed_output, hidden = self.rnn(packed_embedded)
        return self.fc(hidden[-1])


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

 
def collate_fn(batch):
    labels = torch.tensor([item[0] for item in batch], dtype=torch.float)
    texts = [torch.tensor(numericalize_text(item[1])) for item in batch]
    lengths = torch.tensor([len(text) for text in texts])

    # Pad the sequences
    texts_padded = pad_sequence(texts, batch_first=True)
   
    # Sort the batch by descending lengths
    lengths, sort_idx = lengths.sort(descending=True)
    texts_padded = texts_padded[sort_idx]
    labels = labels[sort_idx]
   
    return labels, texts_padded, lengths

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_data, batch_size=64, shuffle=False, collate_fn=collate_fn)

# Function to numericalize the text
def numericalize_text(text):
    return [vocab[token] for token in tokenizer(text)] 

# Training loop with progress bar
epochs = 5
for epoch in range(epochs):
    model.train()
    train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{epochs}')
    for labels, text, lengths in train_bar:
        optimizer.zero_grad()
        predictions = model(text, lengths).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        # Update the progress bar with the loss
        train_bar.set_postfix(loss=loss.item())

# Evaluation loop with progress bar
model.eval()
correct_predictions = 0
total_examples = 0
valid_bar = tqdm(valid_loader, desc='Evaluating')
with torch.no_grad():
    for labels, text, lengths in valid_bar:
        predictions = model(text, lengths).squeeze(1)
        rounded_predictions = torch.round(torch.sigmoid(predictions))
        correct_predictions += (rounded_predictions == labels).sum().item()
        total_examples += labels.size(0)
        # Optionally update the progress bar here if you want to show additional info
        valid_bar.set_postfix(accuracy=100.0 * correct_predictions / total_examples)

accuracy = correct_predictions / total_examples
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
