import nltk
import numpy as np
import sklearn as skl
import os
import tempfile

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# Remove the stop words and punctuations
# Input: String
# Output: list
def remove_stopwords(line):
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(line)
    filtered_sentence = []

    for w in tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


# Lemmatize the given line
# Input: string
# Output: list
def stemmer(line):
    lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
    filtered_sentence = []
    for w in line:
        filtered_sentence.append(lmtzr.lemmatize(w))
    return filtered_sentence




def get_data(stopword=0, stem=0, infrequent=0):
    # Gather info from the files
    neg = open('rt-polaritydata/rt-polarity.neg', encoding='cp1252')
    pos = open('rt-polaritydata/rt-polarity.pos', encoding='cp1252')

    neglines = neg.readlines()
    poslines = pos.readlines()

    X = []
    y = []

    for line in neglines:

        if stopword == 1:
            line = ' '.join(remove_stopwords(line))
            line = line.split(' ')
        if stem == 1:
            line = ' '.join(stemmer(line))
            line = line.split(' ')
        if infrequent == 1:
            print()

        X.append(line)

    for line in poslines:
        if stopword == 1:
            line = ' '.join(remove_stopwords(line))
            line = line.split(' ')
        if stem == 1:
            line = ' '.join(stemmer(line))
            line = line.split(' ')
        if infrequent == 1:
            print()

        X.append(line)

    print(X)


get_data(stopword=1, stem=1)
