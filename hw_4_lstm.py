import numpy as np
import pandas as pd

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

from hw_4_header_lstm import *
from hw_4_preprocessing_lstm import *

def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')

# prepare glove dict
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(
    EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index
nb_words = min(MAX_FEATURES, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words,
                                                        EMBED_SIZE))
for word, i in word_index.items():
    if i >= MAX_FEATURES: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

inp = Input(shape=(MAXLEN,))
x = Embedding(MAX_FEATURES, EMBED_SIZE, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(50, return_sequences=True,
                       dropout=0.1,
                       recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_t, y, batch_size=32, epochs=2, validation_split=0.1);
y_test = model.predict([X_te], batch_size=1024, verbose=1)



sample_submission.to_csv(SUBM_LSTM_FILE, index=False)
