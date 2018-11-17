from sklearn.model_selection import StratifiedKFold
import numpy as np

## for preprocessing
TRAIN_FILE = 'train.csv'
TEST_X_FILE = 'test.csv'
TEST_Y_FILE = 'test_labels.csv'
SUBM = 'sample_submission.csv'
SUBM_LOG_REG_FILE = 'submission_log_reg.csv'

CLASSES = ['toxic', 'severe_toxic', 'obscene',
           'threat', 'insult', 'identity_hate']

WORD_NGRAM_RANGE = (1, 1) # number of words
CHAR_NGRAM_RANGE = (2, 5)
MAX_FEATURES_WORD = 5000
MAX_FEATURES_CHAR = 20000
NUM_SAMPLES_TEST = 50000
NUM_SAMPLES_TRAIN = 50000

lr_list = np.arange(0.01, 0.5, 0.03)
lambda_list = np.arange(0.00001, 0.00009, 0.00002)
b_list = np.arange(0.5, 1, 0.05)

N_SPLITS = 3

# for fmin
MAX_EVALS = 150 # 2

skf = StratifiedKFold(n_splits=N_SPLITS)
