from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from scipy.sparse import hstack

from hw_4_header import *

word_vectorizer = TfidfVectorizer(
    binary=True,
    ngram_range=WORD_NGRAM_RANGE,
    analyzer='word',
    stop_words='english',
    max_features=MAX_FEATURES)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=CHAR_NGRAM_RANGE,
    max_features=MAX_FEATURES)


train = pd.read_csv(TRAIN_FILE).fillna(' ')[:1000]
test = pd.read_csv(TEST_FILE).fillna(' ')[:1000]

train_text = train['comment_text']
test_text = test['comment_text']

all_text = pd.concat([train_text, test_text])
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)


char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})

for class_name in CLASSES:
    train_target = train[class_name]
    classifier = LogisticRegression(C=0.1, solver='sag')

    cv_score = np.mean(cross_val_score(classifier,
                                       train_features,
                                       train_target,
                                       cv=3,
                                       scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('submission.csv', index=False)
