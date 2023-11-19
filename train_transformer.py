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
EPOCHS   = 15
LR       = 1e-3
BATCH_SZ = 64
SPLIT    = 0.9
CONTEXT  = 20

scaler = GradScaler()

# Vocab will take in a [str, str, str] and return [int, int, int]
# Tokenizer takes a string and breaks it up into the appropriate
# tokens. Handles stuff like apostrophes well to. i.e. 
# "It's going to be a great example" turns into:
# ['it', "'", 's', 'going', 'to', 'be', 'a', 'great', 'example']
# Data is the actual data of form [(target, text), ...]
# Note I need this to be globally available for efficiency reasons
vocab, tokenizer, data = get_data_torchtext()

def avg_gradient(model):
    gradient_sum = sum(abs(param.grad.sum().item()) for param in model.parameters() if param.requires_grad)
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
        label  = torch.tensor(label).to(torch.int64)
        return label, vector

def train(dl, model, optim, loss_fn):

    model.train()

    last_batch_time = time()
    epoch_start_time = time()
    avg_loss = 0

    for batch, (labels, texts) in enumerate(dl):

        # Data comes in as [batch_sz, context_size]
        # We need [context_size, batch_sz]
        texts = texts.t().to(DEVICE)
        labels = labels.to(DEVICE)
        
        optim.zero_grad()

        with autocast():
            prediction = model(texts)
            loss = loss_fn(prediction, labels)

        avg_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        if batch % 100 == 0:
            delta = time() - last_batch_time
            delta = timedelta(seconds=delta)
            avg_loss /= 100
            print(f"Batch {batch:4d}/{len(dl):4d} | loss = {avg_loss:.5f} | {delta}")
            avg_loss = 0
            avg_gradient(model)

    epoch_time = timedelta(seconds=time() - epoch_start_time)
    print(f"Epoch took {epoch_time}")


def test(dl, model):

    model.eval()

    # Keep track of average loss for a batch
    avg_loss = 0 
    correct = 0

    print_okay = True
    
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

        if print_okay:
            #print(f"Prediciton: {prediction}")
            print(pred_idx)
            print(f"labels: {labels}")
            print_okay = False

        # Count the number of correct predictions
        correct += torch.sum(pred_idx == labels).item()

        avg_loss += loss.item()

    avg_loss /= len(dl)
    total = len(dl.dataset)

    print(f"Accuracy: {(correct / total)*100:2.2f}")
    print(f"Avg loss: {avg_loss:.5f}")

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
        embed_size=256,       # We have a decent sized vocab so I am selecting a fairly high dim
        num_heads=2,          # Dunno, common practice
        num_layers=1,        # Dunno, common practice
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

    for epoch in range(1, EPOCHS+1):
        print(f"Starting epoch {epoch}")
        train(train_dl, model, optim, loss_fn)
        test(test_dl, model)

