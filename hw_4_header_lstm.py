EMBEDDING_FILE='glove.6B.50d.txt'
TRAIN_DATA_FILE='train.csv'
TEST_DATA_FILE='test.csv'

EMBED_SIZE = 50 # how big is each word vector
MAX_FEATURES = 8000 # how many unique words to
# use (i.e num rows in embedding vector)
MAXLEN = 100 # max number of words in a comment to use

## for preprocessing
TEST_Y_FILE = 'test_labels.csv'
SUBM = 'sample_submission.csv'
SUBM_LSTM_FILE = 'submission_lstm.csv'

CLASSES = ['toxic', 'severe_toxic', 'obscene',
           'threat', 'insult', 'identity_hate']

MAX_FEATURES_WORD = 700
NUM_SAMPLES_TEST = 1000
NUM_SAMPLES_TRAIN = 1000
