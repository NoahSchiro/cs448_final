import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

from utils import load_data, clean_text, additional_cleaning

# Load the dataset
file_path = '../cs448_final/data/data.csv'
df = load_data(file_path)


df['text'] = df['text'].progress_apply(clean_text)
df['text'] = df['text'].progress_apply(additional_cleaning)


# Map labels to integers
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['target'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=42)

# Tokenize and convert text data to numerical representations using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# Build the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.EmbeddingBag(input_size, hidden_size, sparse=True)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded.view(len(x), 1, -1))
        output = self.fc(output.view(len(x), -1))
        output = self.sigmoid(output)
        return output

# Initialize the model
vocabulary_size = X_train.shape[1]
hidden_size = 100
output_size = 1
model = RNNModel(vocabulary_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 5
batch_size = 32

# Convert to DataLoader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) 

# Training loop
for epoch in range(num_epochs):
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        optimizer.zero_grad()
        outputs = model(inputs.long())
        loss = criterion(outputs.squeeze(), labels) 
        loss.backward()
        optimizer.step()

# Evaluation
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test.long())
    predicted_labels = (test_outputs.squeeze() > 0.5).float()
    accuracy = (predicted_labels == y_test).float().mean()

print(f'Accuracy: {accuracy.item()}')

