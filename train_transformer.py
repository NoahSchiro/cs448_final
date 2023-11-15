from src.utils import get_data_torchtext

if __name__=="__main__":

    # Vocab will take in a [str, str, str] and return [int, int, int]
    # Tokenizer takes a string and breaks it up into the appropriate
    # tokens. Handles stuff like apostrophes well to. i.e. 
    # "It's going to be a great example" turns into:
    # ['it', "'", 's', 'going', 'to', 'be', 'a', 'great', 'example']
    vocab, tokenizer = get_data_torchtext()

    print(vocab(tokenizer("Here is an an example string")))

    
