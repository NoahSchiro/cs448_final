from random import shuffle

import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Get data using existing torchtext methods
def get_data_torchtext():

    # Pull in csv data
    path = "./data/data.csv"
    columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(path, header=None, names=columns, encoding='ISO-8859-1')

    # We only care about the text and the label
    df = df[["target", "text"]]

    # Convert to a python object so I don't have to deal with pandas
    # This has form [(target, text), (target, text), ...]
    data = [tuple(x) for x in df.to_numpy()]
    shuffle(data)
    data = data[:int(len(data) * 0.01)]

    # Torchtext tokenizer
    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data):
        for _, text in data:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(data), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    return vocab, tokenizer, data
