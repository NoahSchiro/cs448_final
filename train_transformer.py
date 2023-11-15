from src.utils import get_data_torchtext
from src.transformer import TransformerModel
import torch

if __name__=="__main__":

    # Vocab will take in a [str, str, str] and return [int, int, int]
    # Tokenizer takes a string and breaks it up into the appropriate
    # tokens. Handles stuff like apostrophes well to. i.e. 
    # "It's going to be a great example" turns into:
    # ['it', "'", 's', 'going', 'to', 'be', 'a', 'great', 'example']
    # Data is the actual data of form [(target, text), ...]
    #vocab, tokenizer, data = get_data_torchtext()

    # Create a function that processes text
    #text_fn  = lambda x: vocab(tokenizer(x))

    # Maximum token length is 229
    # Average token length is 16.56

    model = TransformerModel(
        vocab_size=70000,#len(vocab),
        embed_size=256,   # We have a decent sized vocab so I am selecting a fairly high dim
        num_heads=8,      # Dunno, common practice
        num_layers=8,     # Dunno, common practice
        context_size=230, # Determined by max tweet size
        num_classes=3     # Determined by dataset
    )

    batch = 1
    rand = torch.randint(0, 70000, (230, batch))
    print(rand)

    out = model(rand)

    print(out)
