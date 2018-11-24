import pandas as pd
from hw_4_header_nb_svm import *
from sklearn.metrics import roc_auc_score
import hw_4_preprocessing_nb_svm as preprocessing
import numpy as np

LOG_REG_RESULT = 'submission_log_reg_isp.csv'
NB_SVM_RESULT = 'submission_nb_svm.csv'
LSTM_RESULT = 'submission_lstm.csv'
test_y = preprocessing.test_y
shape = len(test_y)

p_log_reg = pd.read_csv(LOG_REG_RESULT) [:shape]
p_nb_svm = pd.read_csv(NB_SVM_RESULT) [:shape]
p_lstm = pd.read_csv(LSTM_RESULT) [:shape]

p_res = p_lstm.copy()
p_res[CLASSES] = (0.5 * p_lstm[CLASSES] +
                  0.0 * p_nb_svm[CLASSES] +
                  0.5 * p_log_reg[CLASSES]) / 2.0

p_res.to_csv('submission.csv', index=False)

scores = []
for i, class_name in enumerate(CLASSES):
    pred_y = p_res[class_name]
    score = roc_auc_score(test_y[class_name], pred_y)
    scores.append(score)
    print("for {}, score {}".format(class_name , score))

print('Total CV score is {}'.format(np.mean(scores)))
