
## for preprocessing
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
CLASSES = ['toxic', 'severe_toxic', 'obscene',
           'threat', 'insult', 'identity_hate']
WORD_NGRAM_RANGE = (1, 1)
CHAR_NGRAM_RANGE = (2, 6)
MAX_FEATURES = 100

## for log_reg
# LR = 0.01
# NUM_ITER = 100
# THRESHOLD = 0.5
# LAMBDA = 0.5
