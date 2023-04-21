from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

def preprocess_text(df):
    # Convert the 'text' column to strings
    df['text'] = df['text'].astype(str)

    # Tokenize the text using TweetTokenizer
    tokenizer = TweetTokenizer()
    df['tokens'] = df['text'].apply(tokenizer.tokenize)

    # Load stop words
    sw = stopwords.words('english')
    # Obtaining Additional Stopwords From nltk
    sw.extend(['from', 'subject', 're', 'edu', 'use'])

    # Remove stop words
    df['tokens'] = [[word for word in t if word not in sw] for t in tqdm(df['tokens'])]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    df['tokens'] = [[lemmatizer.lemmatize(word) for word in t] for t in tqdm(df['tokens'] )]

    # Remove non-word characters
    tokenizers = RegexpTokenizer(r'\w+')
    df['tokens'] = [["".join(tokenizers.tokenize(word)) for word in t 
                   if len(tokenizers.tokenize(word)) > 0]for t in tqdm(df['tokens'])]

    # Combine tokens back into cleaned text
    clean_text = [" ".join(text) for text in tqdm(df['tokens'])]
    df['Clean_text'] = clean_text

    return df