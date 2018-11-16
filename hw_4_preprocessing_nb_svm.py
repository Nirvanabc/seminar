import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from hw_4_header_nb_svm import *

# commented lines can be useful for testing

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()

train = pd.read_csv(TRAIN_FILE) [:NUM_SAMPLES_TRAIN]
test = pd.read_csv(TEST_X_FILE) [:NUM_SAMPLES_TEST]
test_y_all = pd.read_csv(TEST_Y_FILE) [:NUM_SAMPLES_TEST]

# delete samples that don't play in test (they are marked with -1)
test_y = test_y_all[~(test_y_all.toxic == -1)]
test = test[~(test_y_all.toxic == -1)]

train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

vec = TfidfVectorizer(ngram_range=(1,2),
                      tokenizer=tokenize,
                      min_df=3,
                      max_df=0.9,
                      analyzer=ANALYZER,
                      strip_accents='unicode',
                      use_idf=1,
                      smooth_idf=1,
                      sublinear_tf=1,
                      max_features=MAX_FEATURES_WORD)

X = vec.fit_transform(train[COMMENT])
test_X = vec.transform(test[COMMENT])
