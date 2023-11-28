import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import data_tabular
from torchtext.legacy.data import BucketIterator

# Importing util functions
from utils import get_data_torchtext

# Loading data and preprocessing
file_path = "./data/data.csv"
vocab_specials = ["<unk>"]
get_data_torchtext(file_path=file_path, vocab_specials=vocab_specials)

# Defining the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        assert torch.equal(output[-1, :, :], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))

# Tokenizer
tokenizer = get_tokenizer("basic_english")

# Defining a generator function to yield tokens
def yield_tokens(data):
    for _, text in data:
        yield tokenizer(text)

# Building vocabulary
data = data_tabular(
    path=file_path,
    format='csv',
    fields=[('target', None), ('text', None)],
    skip_header=True,
    csv_reader_params={'quotechar': '"', 'quoting': True}
)

vocab = build_vocab_from_iterator(yield_tokens(data), specials=vocab_specials)
vocab.set_default_index(vocab[vocab_specials[0]])

# Splitting the data into training and validation sets
train_data, valid_data = data.split(split_ratio=0.8)

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

# Updating the fields in the dataset with the numericalized text
train_data.fields['text'].numericalize = numericalize_text
valid_data.fields['text'].numericalize = numericalize_text

# Using BucketIterator for efficient batch handling
train_iterator, valid_iterator = BucketIterator.splits(
    (train_data, valid_data),
    batch_size=64,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True
)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    for batch in train_iterator:
        text, text_lengths = batch.text
        optimizer.zero_grad()
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, batch.target)
        loss.backward()
        optimizer.step()

# Evaluation loop
model.eval()
correct_predictions = 0
total_examples = 0

with torch.no_grad():
    for batch in valid_iterator:
        text, text_lengths = batch.text
        predictions = model(text).squeeze(1)
        rounded_predictions = torch.round(torch.sigmoid(predictions))
        correct_predictions += (rounded_predictions == batch.target).sum().item()
        total_examples += batch.target.size(0)

accuracy = correct_predictions / total_examples
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
