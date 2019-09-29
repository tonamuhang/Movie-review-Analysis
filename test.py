import sklearn
import numpy as np
import pandas as pd
import nltk
import os

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier
from sklearn import svm
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import load_files
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

stop_words = set(nltk.corpus.stopwords.words('english'))
tokenized_stop_words = nltk.word_tokenize(' '.join(nltk.corpus.stopwords.words('english')))


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in nltk.word_tokenize(doc)]


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
    for i, line in enumerate(neg):
        if i < 0.9 * length:
            f = open('rt-polaritydata/train/neg/input_%i.data' % i, 'w', encoding='UTF-8')
        else:
            f = open('rt-polaritydata/test/neg/input_%i.data' % i, 'w', encoding='UTF-8')
        f.write(line)
        f.close()
    for i, line in enumerate(pos):
        if i < 0.9 * length:
            f = open('rt-polaritydata/train/pos/input_%i.data' % i, 'w', encoding='UTF-8')
        else:
            f = open('rt-polaritydata/test/pos/input_%i.data' % i, 'w', encoding='UTF-8')
        f.write(line)
        f.close()

    train = load_files('rt-polaritydata/train')
    train_x, train_y = train.data, train.target

    test = load_files('rt-polaritydata/test')
    test_x, test_y = test.data, test.target

    # TODO
    # cvec = CountVectorizer(ngram_range=(1, 1), stop_words=tokenized_stop_words,
    # min_df=infreq*min, tokenizer=LemmaTokenizer())
    cvec = CountVectorizer(ngram_range=(1, 1), min_df=3)

    train_x = cvec.fit(train_x).transform(train_x)
    test_x = cvec.transform(test_x)

    # logistic regression
    parameter = {'C': [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(), parameter, cv=5)
    grid.fit(train_x, train_y)

    print("LR accuracy: {:.4f}".format(grid.best_score_))

    # SVM
    parameter = {'C': [0.1, 1, 10]}
    grid = GridSearchCV(svm.LinearSVC(), parameter, cv=5)
    grid.fit(train_x, train_y)
    print("SVM accuracy: {:.4f}".format(grid.best_score_))

    # Naive Bayes
    parameter = {'alpha': [0, 1, 2, 3]}
    df_train_x = pd.DataFrame(train_x.todense(), columns=cvec.get_feature_names())
    grid = GridSearchCV(MultinomialNB(), parameter, cv = 5)
    grid.fit(df_train_x, train_y)
    print("NB accuracy: {:.4f}".format(grid.best_score_))

    clf = MultinomialNB(alpha=grid.best_params_.get('alpha'))
    df_test = pd.DataFrame(test_x.todense(), columns=cvec.get_feature_names())
    clf.fit(df_train_x, train_y)
    predicted = clf.predict(df_test)
    results = confusion_matrix(test_y, predicted)
    print("Confusion matrix: ")
    print(results)
    plt.imshow(results, cmap='binary', interpolation='None')
    plt.show()

    # Dummy classifier random
    dummy_classifier = DummyClassifier(strategy="uniform")
    dummy_classifier.fit(train_x, train_y)
    predicted = dummy_classifier.predict(test_x)
    print("Dummy accuracy: ", accuracy_score(test_y, predicted))




get_data()