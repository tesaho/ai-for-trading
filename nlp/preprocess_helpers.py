"""
Pre-process and clean SEC docs

"""

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from bs4 import BeautifulSoup

# download nlp corpus
nltk.download('stopwords')
nltk.download('wordnet')

def remove_html_tags(text):
    return BeautifulSoup(text, 'html.parser').get_text()

def clean_text(text):
    text = text.lower()
    return remove_html_tags(text)

def lemmatize_words(words, lemmatizer = WordNetLemmatizer()):
    """
    lemmatize words
    params:
        words: list of words (str)
    return:
        list of lemmatized words
    """

    return [lemmatizer.lemmatize(word, pos='v') for word in words]


def lemmatize_stopwords(stopwords, lemmatizer = WordNetLemmatizer()):
    """
    lemmatize stop words
    params:
        stopwords: list of stopwords(str)
    return:
        list of lemmatized stopwrods
    """

    return lemmatize_words(stopwords.words('english'), lemmatizer = lemmatizer)
