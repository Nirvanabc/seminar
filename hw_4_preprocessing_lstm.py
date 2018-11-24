import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from hw_4_header_lstm import *

train = pd.read_csv(TRAIN_DATA_FILE) # [:NUM_SAMPLES_TEST]
test = pd.read_csv(TEST_DATA_FILE) # [:NUM_SAMPLES_TEST]

test_y_all = pd.read_csv(TEST_Y_FILE) # [:NUM_SAMPLES_TEST]
test_y = test_y_all[~(test_y_all.toxic == -1)]
test = test[~(test_y_all.toxic == -1)]

list_sentences_train = train["comment_text"].fillna("_na_").values
y = train[CLASSES].values
list_sentences_test = test["comment_text"].fillna("_na_").values

tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=MAXLEN)
X_te = pad_sequences(list_tokenized_test, maxlen=MAXLEN)

sample_submission = pd.read_csv(SUBM) [:NUM_SAMPLES_TEST]
sample_submission = sample_submission[~(test_y_all.toxic == -1)]
