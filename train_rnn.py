import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from utils import get_data_torchtext
from tqdm import tqdm
import matplotlib.pyplot as plt


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
train_losses = []
valid_losses = []
train_accuracies = [] 
accuracies = []

for epoch in range(epochs):
    model.train()
    train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{epochs}')
    epoch_train_losses = []
    correct_predictions_train = 0  
    total_examples_train = 0  
    for labels, text, lengths in train_bar:
        optimizer.zero_grad()
        predictions = model(text, lengths).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_train_losses.append(loss.item())
        train_bar.set_postfix(loss=loss.item())

        # Calc training accuracy during training loop
        rounded_predictions_train = torch.round(torch.sigmoid(predictions))
        correct_predictions_train += (rounded_predictions_train == labels).sum().item()
        total_examples_train += labels.size(0)
 

    # Calc training loss and accuracy for the epoch
    avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
    train_losses.append(avg_train_loss)
    accuracy_train = correct_predictions_train / total_examples_train
    train_accuracies.append(accuracy_train)

    # Validation loop with progress bar
    model.eval()
    correct_predictions = 0
    total_examples = 0
    epoch_valid_losses = []
    valid_bar = tqdm(valid_loader, desc='Evaluating')

    with torch.no_grad():
        for labels, text, lengths in valid_bar:
            predictions = model(text, lengths).squeeze(1)
            loss = criterion(predictions, labels)
            epoch_valid_losses.append(loss.item())
            rounded_predictions = torch.round(torch.sigmoid(predictions))
            correct_predictions += (rounded_predictions == labels).sum().item()
            total_examples += labels.size(0)
            valid_bar.set_postfix(accuracy=100.0 * correct_predictions / total_examples)

    # Calc validation loss and accuracy for the epoch
    avg_valid_loss = sum(epoch_valid_losses) / len(epoch_valid_losses)
    valid_losses.append(avg_valid_loss)
    accuracy = correct_predictions / total_examples
    accuracies.append(accuracy)

# Plotting accuracy vs epochs with training accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), accuracies, label='Validation Accuracy')
plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')  
plt.title('Validation and Training Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./results/accuracy_vs_epochs.png')
plt.show()
# Plotting training and validation loss vs epochs
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), valid_losses, label='Validation Loss')
plt.title('Training and Validation Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./results/loss_vs_epochs.png')
plt.show()