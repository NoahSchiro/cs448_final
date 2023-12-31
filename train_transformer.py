from time import time
from datetime import timedelta
from tqdm import tqdm

from src.utils import get_data_torchtext
from src.transformer import TransformerModel, SimpleTextClassifier
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# AMP optimizations
from torch.cuda.amp import GradScaler, autocast

DEVICE   = torch.device("cuda") if torch.cuda.is_available() else torch.device("CPU")
EPOCHS   = 1 # Model tends to overfit after just 1 epoch!
LR       = 1e-3
BATCH_SZ = 64
SPLIT    = 0.9
CONTEXT  = 50

# TODO: Add training and validation accuracy history (added, need to test on main PC)
# TODO: Add training and validation loss history (added, need to test on main PC)
# TODO: Add model inference on new text (added, need to test on main PC)
# this means I also need to save the tokenizer and vocab

scaler = GradScaler()
train_loss_history = []
val_loss_history   = []
train_acc_history  = [50.]
val_acc_history    = []

# Vocab will take in a [str, str, str] and return [int, int, int]
# Tokenizer takes a string and breaks it up into the appropriate
# tokens. Handles stuff like apostrophes well to. i.e. 
# "It's going to be a great example" turns into:
# ['it', "'", 's', 'going', 'to', 'be', 'a', 'great', 'example']
# Data is the actual data of form [(target, text), ...]
# Note I need this to be globally available for efficiency reasons
vocab, tokenizer, data = get_data_torchtext()
torch.save(vocab, "./results/vocab.pth")
torch.save(tokenizer, "./results/tokenizer.pth")

# For debugging
def avg_gradient(model):
    gradient_sum    = sum(abs(param.grad.sum().item()) for param in model.parameters() if param.requires_grad)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Avg gradient: {gradient_sum / parameter_count}")

class TransformerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def text_preprocessing(self, text):

        tokenized = tokenizer(text)

        if len(tokenized) < CONTEXT:
            # Append the "unkown" token till we get to CONTEXT 
            while len(tokenized) < CONTEXT:
                tokenized.append("<unk>")

        # Or slice the list down
        else:
            tokenized = tokenized[:CONTEXT]

        # 1.Turn tokens into ints
        # 2.Turn list of ints into tensor
        # 3.Turn it into a column tensor
        return torch.tensor(vocab(tokenized))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label, text = self.data[index]
        vector = self.text_preprocessing(text)
        # Weirdly, the labels are
        # 0 -> Negative
        # 4 -> Positive.
        label /= 4
        label = torch.tensor(label).to(torch.int64)
        return label, vector

def train(dl, model, optim, loss_fn):

    model.train()

    last_batch_time = time()
    epoch_start_time = time()
    avg_loss = 0
    correct = 0
    total = 0

    for batch, (labels, texts) in enumerate(dl):

        # Data comes in as [batch_sz, context_size]
        # We need [context_size, batch_sz]
        texts = texts.t().to(DEVICE)
        labels = labels.to(DEVICE)
        
        optim.zero_grad()

        with autocast():
            prediction = model(texts)
            loss = loss_fn(prediction, labels)
        
        # Count the number of correct predictions
        pred_idx = torch.argmax(prediction, dim=1)
        correct += torch.sum(pred_idx == labels).item()
        total   += BATCH_SZ

        avg_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        if batch % 100 == 0 and batch != 0:
            delta = time() - last_batch_time
            delta = timedelta(seconds=delta)
            avg_loss /= 100
            train_loss_history.append(avg_loss)
            print(f"Batch {batch:4d}/{len(dl):4d} | loss = {avg_loss:.5f} | {delta}")
            avg_loss = 0

            acc = correct / total
            acc *= 100
            train_acc_history.append(acc)
            correct = 0
            total = 0

            #avg_gradient(model) debugging code

    epoch_time = timedelta(seconds=time() - epoch_start_time)
    print(f"Epoch took {epoch_time}")


def test(dl, model):

    model.eval()

    # Keep track of average loss for a batch
    avg_loss = 0 
    correct = 0
    
    print("Testing...")
    for (labels, texts) in tqdm(dl):
 
        # Data comes in as [batch_sz, context_size]
        # We need [context_size, batch_sz]
        texts = texts.t().to(DEVICE)
        labels = labels.to(DEVICE)

        with autocast():
            prediction = model(texts)
            loss = loss_fn(prediction, labels)
        
        pred_idx = torch.argmax(prediction, dim=1)

        # Count the number of correct predictions
        correct += torch.sum(pred_idx == labels).item()

        avg_loss += loss.item()

    avg_loss /= len(dl)
    val_loss_history.append(avg_loss)

    total = len(dl.dataset)

    accuracy = (correct / total) * 100
    val_acc_history.append(accuracy)

    print(f"Accuracy: {accuracy:2.2f}")
    print(f"Avg loss: {avg_loss:.5f}")

    return accuracy

def generate_graphs():

    import matplotlib.pyplot as plt
    import numpy as np

    total_batches = len(train_loss_history)
    epochs = np.linspace(0, EPOCHS, num=total_batches)

    # Training / validation graph
    plt.plot(epochs, train_loss_history, 'b', label='Training Loss')
    plt.plot(np.arange(0, EPOCHS+1), val_loss_history, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig("./results/loss.png")
    plt.clf()
    
    total_batches = len(train_acc_history)
    epochs = np.linspace(0, EPOCHS, num=total_batches)

    # Accuracy graph
    plt.plot(epochs, train_acc_history, 'b', label='Train accuracy')
    plt.plot(np.arange(0, EPOCHS+1), val_acc_history, 'r', label='Validation accuracy')
    plt.title('Changes in accuracy over time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig("./results/accuracy.png")
    plt.clf()



if __name__=="__main__":
    
    # Maximum token length is 229
    # Average token length is 16.56

    print("Data loaded...")
    print(f"Dataset size: {len(data)}")
    print(f"Vocab size: {len(vocab)}")
    print("Working on preprocessing...")

    # Train test split
    split_idx = int(len(data) * SPLIT)
    train_lst, test_lst = data[:split_idx], data[split_idx:]
    del data

    # Convert to pytorch datasets
    train_ds = TransformerDataset(train_lst)
    test_ds  = TransformerDataset(test_lst)

    # Convert to DL
    # Add pin_mem and num_workers when we get to optimization
    train_dl = DataLoader(train_ds, batch_size=BATCH_SZ, shuffle=True, pin_memory=True, num_workers=12)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SZ, shuffle=True, pin_memory=True, num_workers=12)

    model = TransformerModel(
        vocab_size=len(vocab),
        embed_size=264,       # We have a decent sized vocab so I am selecting a fairly high dim
        num_heads=12,         # Dunno, common practice
        num_layers=1,         # Anything more than 1 and we run into a vanishing gradient
        context_size=CONTEXT, # Determined by max tweet size
        num_classes=2         # Determined by dataset
    ).to(DEVICE)
    # model = SimpleTextClassifier(
    #     vocab_size=len(vocab),
    #     embed_size=256,
    #     hidden_size=64,
    #     num_classes=2
    # ).to(DEVICE)

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_acc = 0.0

    # Pretraining test
    test(test_dl, model)

    for epoch in range(1, EPOCHS+1):
        print(f"Starting epoch {epoch}")
        train(train_dl, model, optim, loss_fn)
        acc = test(test_dl, model)

        # Save the best performing model
        if acc > best_acc:
            torch.save(model.state_dict(), "./results/transformer_best.pth")
            best_acc = acc
        print(f"Best acc is now {best_acc:.2f}%")

    # Report metrics
    generate_graphs()
