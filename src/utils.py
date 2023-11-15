# Data preprocessing stuff and loading in the data will go here.
# 1. all lower case
# 2. remove punctuation besides periods. Convert periods to a token like <PERIOD>
# 3. Start of text and end of text token <S> <E>
# 4. convert numbers to a number token <NUM>
# 5. More stuff as we think of is


import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm #for a progress bar... removing stop words take long
from nltk.stem import WordNetLemmatizer # issues unzipping wordnet package...


# Data preprocessing
def clean_text(text):
    # Convert to lowercase
    text = text.lower()

     # Expand contractions
    contractions = {"can't": "cannot", "won't": "will not", "I'm": "I am", "it's": "it is", "didn't": "did not"}
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # Remove special characters and symbols
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    
    # Remove punctuation (excluding periods)
    text = re.sub(r'[^\w\s.]', '', text)
    
    # Replace periods with a token
    text = re.sub(r'\.', ' <PERIOD> ', text)
    
    # Add start and end of text tokens
    text = '<S> ' + text + ' <E>'
    
    # Convert numbers to a number token
    text = re.sub(r'\b\d+\b', ' <NUM> ', text)
    
    return text

def additional_cleaning(text):
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Join the tokens back into a sentence
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def removed_url(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove user handles
    text = re.sub(r'@[\w_]+', '', text)
    
    return text

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text

def get_data():
    # Load the dataset
    file_path = './data/data.csv'
    columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(file_path, header=None, names=columns, encoding='ISO-8859-1')
    
    # Vectorized cleaning
    tqdm.pandas()
    df['text'] = df['text'].apply(clean_text)
    df['text'] = df['text'].apply(additional_cleaning)
    df['text'] = df['text'].apply(removed_url)
    df['text'] = df['text'].apply(lemmatize_text)

    return df

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

    # Torchtext tokenizer
    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data):
        for _, text in data:
            yield tokenizer(text)


    vocab = build_vocab_from_iterator(yield_tokens(data), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    return vocab, tokenizer
