
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


# Load the dataset
file_path = '../cs448_final/data/data.csv'
columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv(file_path, header=None, names=columns, encoding='ISO-8859-1')

def load_data(file_path):
    columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(file_path, header=None, names=columns, encoding='ISO-8859-1')
    return df

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
# Vectorized cleaning
tqdm.pandas()
df['text'] = df['text'].apply(clean_text)
df['text'] = df['text'].apply(additional_cleaning)
df['text'] = df['text'].apply(removed_url)
df['text'] = df['text'].apply(lemmatize_text) 

# Display the preprocessed data
print(df.head())


"""This code cleans the dataset by:
Removes stopwords
Expanding contractions.
Removing special characters and symbols.
Removing URLs and user handles.
Lemmatization using the WordNetLemmatizer from NLTK
In additon to what Noah mentioned above.""" 



