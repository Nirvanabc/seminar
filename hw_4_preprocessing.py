from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from scipy.sparse import hstack, coo_matrix
from hw_4_log_reg import *

from hw_4_header import *

word_vectorizer = TfidfVectorizer(
    binary=True,
    ngram_range=WORD_NGRAM_RANGE,
    analyzer='word',
    stop_words='english',
    max_features=MAX_FEATURES_WORD)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=CHAR_NGRAM_RANGE,
    max_features=MAX_FEATURES_CHAR)

train = pd.read_csv(TRAIN_FILE).fillna(' ')[:NUM_SAMPLES_TRAIN]
test_X = pd.read_csv(TEST_X_FILE).fillna(' ')[:NUM_SAMPLES_TEST]
test_y_all = pd.read_csv(TEST_Y_FILE)[:NUM_SAMPLES_TEST]
test_y = test_y_all[~(test_y_all.toxic == -1)]
test_X = test_X[~(test_y_all.toxic == -1)]

train_text = train['comment_text']
test_text = test_X['comment_text']

all_text = pd.concat([train_text, test_text])
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)


char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])


def prepare_train_X():
    data_len = train_features.shape[0]
    X = train_features.toarray()
    X = np.concatenate([np.ones((data_len, 1)), X], axis=1)
    return X


def prepare_test_X():
    # return prepare_train_X()
    data_len = test_features.shape[0]
    X = test_features.toarray()
    X = np.concatenate([np.ones((data_len, 1)), X], axis=1)
    return X
