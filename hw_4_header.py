from sklearn.model_selection import StratifiedKFold
import numpy as np

## for preprocessing
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
CLASSES = ['toxic', 'severe_toxic', 'obscene',
           'threat', 'insult', 'identity_hate']

WORD_NGRAM_RANGE = (1, 1) # number of words
CHAR_NGRAM_RANGE = (2, 6)
MAX_FEATURES_WORD = 3000
MAX_FEATURES_CHAR = 10000
NUM_SAMPLES = 10000

lr_list = np.arange(0.01, 0.5, 0.05)
lambda_list = np.arange(0.001, 0.01, 0.005)
b_list = np.arange(0.1, 1, 0.01)

N_SPLITS = 3

# for fmin
MAX_EVALS = 10 # 150

skf = StratifiedKFold(n_splits=N_SPLITS)
