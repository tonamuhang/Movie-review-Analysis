import sklearn
import numpy as np
import pandas as pd
import nltk
import os

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier
from sklearn import svm
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import load_files
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

stop_words = set(nltk.corpus.stopwords.words('english'))
tokenized_stop_words = nltk.word_tokenize(' '.join(nltk.corpus.stopwords.words('english')))

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stemmer = nltk.stem.PorterStemmer()

    def __call__(self, articles):
        return [self._stem(self.wnl.lemmatize(t)) for t in nltk.word_tokenize(articles)]

    def _stem(self, token):
        if token in stop_words:
            return token
        return self.stemmer.stem(token)


def get_data():

    # TODO: Revive the code before submitting
    if not os.path.exists('rt-polaritydata/train/neg/'):
        os.makedirs('rt-polaritydata/train/neg/')
    if not os.path.exists('rt-polaritydata/train/pos/'):
        os.makedirs('rt-polaritydata/train/pos/')
    if not os.path.exists('rt-polaritydata/test/neg/'):
        os.makedirs('rt-polaritydata/test/neg/')
    if not os.path.exists('rt-polaritydata/test/pos/'):
        os.makedirs('rt-polaritydata/test/pos/')

    neg = open('rt-polaritydata/rt-polarity.neg', encoding='cp1252')
    length = len(neg.readlines())
    neg.close()

    neg = open('rt-polaritydata/rt-polarity.neg', encoding='cp1252')
    pos = open('rt-polaritydata/rt-polarity.pos', encoding='cp1252')

    # TODO: REVIVE BEFORE SUBMITTING
    # for i, line in enumerate(neg):
    #     if i < 0.8 * length:
    #         f = open('rt-polaritydata/train/neg/input_%i.data' % i, 'w', encoding='UTF-8')
    #     else:
    #         f = open('rt-polaritydata/test/neg/input_%i.data' % i, 'w', encoding='UTF-8')
    #     f.write(line)
    #     f.close()
    # for i, line in enumerate(pos):
    #     if i < 0.8 * length:
    #         f = open('rt-polaritydata/train/pos/input_%i.data' % i, 'w', encoding='UTF-8')
    #     else:
    #         f = open('rt-polaritydata/test/pos/input_%i.data' % i, 'w', encoding='UTF-8')
    #     f.write(line)
    #     f.close()

    train = load_files('rt-polaritydata/train')
    train_x, train_y = train.data, train.target

    test = load_files('rt-polaritydata/test')
    test_x, test_y = test.data, test.target

    # TODO
    # cvec = CountVectorizer(ngram_range=(1, 1), stop_words=tokenized_stop_words,
    # min_df=infreq*min, tokenizer=LemmaTokenizer())
    cvec = CountVectorizer(ngram_range=(1, 1), min_df=0)

    train_x = cvec.fit(train_x).transform(train_x)
    test_x = cvec.transform(test_x)

    # logistic regression
    parameter = {'C': [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(), parameter, cv=5)
    grid.fit(train_x, train_y)

    print("LR accuracy: {:.2f}".format(grid.best_score_))

    # SVM
    clf = svm.SVC(kernel= 'linear', C = 1)
    clf.fit(train_x, train_y)
    predicted = clf.predict(test_x)
    print("SVM Accuracy: ", accuracy_score(test_y, predicted))

    # Naive Bayes
    clf = GaussianNB()
    df_train_x = pd.DataFrame(train_x.todense(), columns=cvec.get_feature_names())
    df_test = pd.DataFrame(test_x.todense(), columns=cvec.get_feature_names())
    clf.fit(df_train_x, train_y)
    predicted = clf.predict(df_test)
    print("NB Accuracy: ", accuracy_score(test_y, predicted))

    # Dummy classifier random
    dummy_classifier = DummyClassifier(strategy="uniform")
    dummy_classifier.fit(train_x, train_y)
    predicted = dummy_classifier.predict(test_x)
    print("Dummy accuracy: ", accuracy_score(test_y, predicted))

get_data()