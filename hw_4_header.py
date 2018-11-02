# with max_f_wors = 3000, max_f_char = 10000 and num_samples = 2000 it

from sklearn.model_selection import StratifiedKFold
import numpy as np

## for preprocessing
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
CLASSES = ['toxic', 'severe_toxic', 'obscene',
           'threat', 'insult', 'identity_hate']
WORD_NGRAM_RANGE = (1, 1)
CHAR_NGRAM_RANGE = (2, 6)
MAX_FEATURES_WORD = 300
MAX_FEATURES_CHAR = 1000
NUM_SAMPLES = 2000

lr_list = np.arange(0.1, 1, 0.1)
lambda_list = np.arange(0.1, 1, 0.1)
b_list = np.arange(0, 1, 0.1)


N_SPLITS = 5

# for fmin
MAX_EVALS = 50 # 150

skf = StratifiedKFold(n_splits=N_SPLITS)

## for log_reg
# LR = 0.01
# NUM_ITER = 100
# THRESHOLD = 0.5
# LAMBDA = 0.5
