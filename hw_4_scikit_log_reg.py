from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd

import hw_4_preprocessing_log_reg as preprocessing
from hw_4_header_log_reg import *

X = preprocessing.prepare_train_X()
X_test = preprocessing.prepare_test_X()
subm = pd.read_csv(SUBM)


preds_test = np.zeros((len(X_test), len(CLASSES)))
preds_train = np.zeros((len(X), len(CLASSES)))

scores = []
for i, class_name in enumerate(CLASSES):
    # if you don't transfer it to array, you will have a mistake in fit as
    # lines will be counted starting not from zero
    y = np.array(preprocessing.train[class_name])
    y_test = np.array(preprocessing.test_y[class_name])    

    model = LogisticRegression(C=0.1, solver='sag')
    cv_score = np.mean(cross_val_score(model,
                                       X,
                                       y,
                                       cv=3,
                                       scoring='roc_auc'))
    scores.append(cv_score)
    print('score for class {} is {:.4f}'.format(class_name, cv_score))
    model.fit(X, y)
    preds[:,i] = model.predict_proba(X_test)[:, 1]
    

print('score {:.4}'.format(np.mean(scores)))
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds,
                                             columns=CLASSES)],
                       axis=1)
submission.to_csv(SUBM_LOG_REG_FILE, index=False)
