import pandas as pd
from hw_4_header_nb_svm import *
from sklearn.metrics import roc_auc_score
import hw_4_preprocessing_nb_svm as preprocessing

LOG_REG_RESULT = 'submission_log_reg.csv'
NB_SVM_RESULT = 'submission_nb_svm.csv'
test_y = preprocessing.test_y
shape = len(test_y)

p_lstm = pd.read_csv(LOG_REG_RESULT) [:shape]
p_nbsvm = pd.read_csv(NB_SVM_RESULT) [:shape]

p_res = p_lstm.copy()
p_res[CLASSES] = (p_lstm[CLASSES] + p_lstm[CLASSES]) / 2.0

p_res.to_csv('submission.csv', index=False)



scores = []
for i, class_name in enumerate(CLASSES):
    y_pred = p_res[class_name]
    m = len(y_pred)
    p = np.zeros(m)
    score = roc_auc_score(test_y[class_name], y_pred)
    scores.append(score)
    print("for {}, score {}".format(class_name , score))

print('Total CV score is {}'.format(np.mean(scores)))
