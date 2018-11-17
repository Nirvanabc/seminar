## for preprocessing
TRAIN_FILE = 'train.csv'
TEST_X_FILE = 'test.csv'
TEST_Y_FILE = 'test_labels.csv'
SUBM = 'sample_submission.csv'
SUBM_NB_SVM_FILE = 'submission_nb_svm.csv'

CLASSES = ['toxic', 'severe_toxic', 'obscene',
           'threat', 'insult', 'identity_hate']

COMMENT = 'comment_text'
ANALYZER = 'word'

WORD_NGRAM_RANGE = (1, 3) # number of words
CHAR_NGRAM_RANGE = (2, 7)
MAX_FEATURES_WORD = 70000
NUM_SAMPLES_TEST = 20000
NUM_SAMPLES_TRAIN = 20000

C = 4
MAX_ITER = 100
