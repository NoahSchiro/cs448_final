import torch
from torch import nn
from utils import get_data_torchtext
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(SimpleRNNModel, self).__init__()

        self.embedding = nn.EmbeddingBag(vocab_size, embed_size, sparse=True)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        output = self.fc(output[:, -1, :])  # Using the output from the last time step
        return output

# Processing data
vocab, tokenizer, data = get_data_torchtext()

# Map tokens to indices
token_to_index = {token: idx for idx, token in enumerate(vocab.get_itos())}
max_sequence_length = max(len(tokenizer(text)) for _, text in data)
data_indices = torch.tensor(
    [[token_to_index[token] for token in tokenizer(text)] + [0] * (max_sequence_length - len(tokenizer(text))) for label, text in data],
    dtype=torch.long
)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data_indices, [label for label, text in data], test_size=0.2, random_state=42)

# Parameters
vocab_size = len(vocab)
embed_size = 100
hidden_size = 50
num_classes = 2
num_epochs = 5
batch_size = 32
learning_rate = 0.001

# Model, loss, and optimizer
model = SimpleRNNModel(vocab_size, embed_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Converting to DataLoader
train_data = TensorDataset(X_train, torch.tensor(y_train, dtype=torch.long))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Training loop
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluation
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    predicted_labels = torch.argmax(test_outputs, dim=1)
    accuracy = (predicted_labels == torch.tensor(y_test, dtype=torch.long)).float().mean()

print(f'Accuracy: {accuracy.item()}')
