from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from hw_4_header import *
import hw_4_preprocessing_nb_svm as preprocessing

x = preprocessing.X
test_x = preprocessing.test_X
test_y = preprocessing.test_y
train = preprocessing.train

# x = sparse.csr_matrix(preprocessing.train_features)
# test_x = sparse.csr_matrix(preprocessing.test_features)

class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=C, solver='lbfgs', max_iter=MAX_ITER):
        self.C = C
        self.solver = solver
        self.max_iter = max_iter

    def predict(self, x):
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)
        
        y = y.values
        self._r = np.log(pr(x,1,y) / pr(x,0,y))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C,
                                       solver=self.solver,
                                       max_iter=self.max_iter).fit(
                                           x_nb,
                                           y)

scores = []
for class_name in CLASSES:
    model = NbSvmClassifier(C=4)
    model.fit(x, train[class_name])
    y_pred = model.predict(test_x)
    score = roc_auc_score(test_y[class_name], y_pred)
    scores.append(score)
    print("for {}, score {}".format(class_name , score))

print('Total CV score is {}'.format(np.mean(scores)))
