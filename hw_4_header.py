from sklearn.model_selection import StratifiedKFold
import numpy as np

## for preprocessing
TRAIN_FILE = 'train.csv'
TEST_X_FILE = 'test.csv'
TEST_Y_FILE = 'test_labels.csv'
CLASSES = ['toxic', 'severe_toxic', 'obscene',
           'threat', 'insult', 'identity_hate']

WORD_NGRAM_RANGE = (1, 1) # number of words
CHAR_NGRAM_RANGE = (2, 6)
MAX_FEATURES_WORD = 5000
MAX_FEATURES_CHAR = 10000
NUM_SAMPLES_TEST = 7000
NUM_SAMPLES_TRAIN = 7000

lr_list = np.arange(0.01, 0.5, 0.03)
lambda_list = np.arange(0.00001, 0.00009, 0.00002)
b_list = np.arange(0.5, 1, 0.05)

N_SPLITS = 3

# for fmin
MAX_EVALS = 50 # 150

skf = StratifiedKFold(n_splits=N_SPLITS)
