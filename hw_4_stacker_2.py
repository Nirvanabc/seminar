# 153165 submission_log_reg_isp.csv
# 63979 submission_lstm.csv
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np

from hw_4_header_nb_svm import *

LOG_REG_RESULT = 'submission_log_reg_isp.csv'
NB_SVM_RESULT = 'submission_nb_svm.csv'
LSTM_RESULT = 'submission_lstm.csv'
SHAPE = 8670
SHAPE_TEST = SHAPE//10

test_y_all = pd.read_csv(TEST_Y_FILE)

def prepare_metadata(data_file, need_to_filter):
    res = pd.read_csv(data_file)
    if need_to_filter:
        res = res[~(test_y_all.toxic == -1)]
        res = res.reset_index(drop=True)
    return res


# y_test = prepare_metadata(TEST_Y_FILE, True)            [:SHAPE_TEST]
# p_log_reg_test = prepare_metadata(LOG_REG_RESULT, True) [:SHAPE_TEST]
# p_nb_svm_test = prepare_metadata(NB_SVM_RESULT, True)   [:SHAPE_TEST]
# p_lstm_test = prepare_metadata(LSTM_RESULT, False)      [:SHAPE_TEST]
# 
# y_train = prepare_metadata(TEST_Y_FILE, True)            [SHAPE_TEST:SHAPE]
# p_log_reg_train = prepare_metadata(LOG_REG_RESULT, True) [SHAPE_TEST:SHAPE]
# p_nb_svm_train = prepare_metadata(NB_SVM_RESULT, True)   [SHAPE_TEST:SHAPE]
# p_lstm_train = prepare_metadata(LSTM_RESULT, False)      [SHAPE_TEST:SHAPE]
# 
# X_test = pd.concat([p_log_reg_test, p_lstm_test], axis=1)
# X_train = pd.concat([p_log_reg_train, p_lstm_train], axis=1)

y = prepare_metadata(TEST_Y_FILE, True) [:SHAPE]


scores = []
#$ for i, class_name in enumerate(CLASSES):
class_name = 'toxic'

# p_res = X_train[class_name]
# y = y_train[class_name]

model = LogisticRegression(C=0.1, solver='sag')
cv_score = np.mean(cross_val_score(model,
                                   p_res,
                                   y,
                                   cv=3,
                                   scoring='roc_auc'))
print(roc_auc_score(y_test, cv_test)
print("for {}, score {}".format(class_name , score))
# 
# print('Total CV score is {}'.format(np.mean(scores)))

