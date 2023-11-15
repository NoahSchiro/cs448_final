from src.utils import get_data_torchtext

if __name__=="__main__":

    # Vocab will take in a [str, str, str] and return [int, int, int]
    # Tokenizer takes a string and breaks it up into the appropriate
    # tokens. Handles stuff like apostrophes well to. i.e. 
    # "It's going to be a great example" turns into:
    # ['it', "'", 's', 'going', 'to', 'be', 'a', 'great', 'example']
    # Data is the actual data of form [(target, text), ...]
    vocab, tokenizer, data = get_data_torchtext()

    # Create a function that processes text
    text_fn  = lambda x: vocab(tokenizer(x))

    print(text_fn("Here is an an example string"))

    
