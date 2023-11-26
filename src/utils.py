
import pandas as pd
from random import sample
from sklearn.preprocessing import LabelEncoder
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def get_data_torchtext(file_path="./data/data.csv", sample_fraction=0.4, vocab_specials=["<unk>"]):
    # Read CSV data
    columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(file_path, header=None, names=columns, encoding='ISO-8859-1')

    # Extract relevant columns
    df = df[["target", "text"]]

    # Label encoding for target
    label_encoder = LabelEncoder() 
    df['target'] = label_encoder.fit_transform(df['target'])

    # Convert to a list of tuples
    data = list(zip(df["target"].tolist(), df["text"].tolist()))

    # Shuffle and sample the data
    data = sample(data, int(len(data) * sample_fraction))

    # Torchtext tokenizer
    tokenizer = get_tokenizer("basic_english")

    # Define a generator function to yield tokens
    def yield_tokens(data):
        for _, text in data:
            yield tokenizer(text)

    # Build vocabulary
    vocab = build_vocab_from_iterator(yield_tokens(data), specials=vocab_specials)
    vocab.set_default_index(vocab[vocab_specials[0]])

    return vocab, tokenizer, data 
