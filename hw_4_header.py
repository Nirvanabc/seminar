from sklearn.model_selection import StratifiedKFold
import numpy as np

## for preprocessing
TRAIN_FILE = 'train.csv'
TEST_X_FILE = 'test.csv'
TEST_Y_FILE = 'test_labels.csv'
CLASSES = ['toxic', 'severe_toxic', 'obscene',
           'threat', 'insult', 'identity_hate']

COMMENT = 'comment_text'

WORD_NGRAM_RANGE = (1, 1) # number of words
CHAR_NGRAM_RANGE = (2, 7)
MAX_FEATURES_WORD = 10000
MAX_FEATURES_CHAR = 50000
NUM_SAMPLES_TEST = 40000
NUM_SAMPLES_TRAIN = 40000

C = 4
MAX_ITER = 1000
