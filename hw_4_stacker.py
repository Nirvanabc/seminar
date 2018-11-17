import pandas as pd
import numpy as np
import re
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import hw_4_preprocessing_log_reg as preprocessing
from hw_4_header_log_reg import *

X_train = preprocessing.prepare_train_X()
train = preprocessing.train
X_test = preprocessing.prepare_test_X()
sub = pd.read_csv(SUBM)[:len(X_test)]

stacker = lgb.LGBMClassifier(max_depth=3,
                             metric="auc",
                             n_estimators=50,
                             num_leaves=10,
                             boosting_type="gbdt",
                             learning_rate=0.1,
                             feature_fraction=0.45,
                             colsample_bytree=0.45,
                             bagging_fraction=0.8,
                             bagging_freq=5,
                             reg_lambda=0.2)

# Fit and submit    
scores = []
for label in CLASSES:
    print(label)
    score = cross_val_score(stacker,
                            X_train,
                            train[label],
                            cv=5,
                            scoring='roc_auc')
    print("AUC:", score)
    scores.append(np.mean(score))
    stacker.fit(X_train, train[label])
    print (stacker.predict_proba(X_test)[:,1].shape)
    print(sub[label].shape)
    sub[label] = stacker.predict_proba(X_test)[:,1]
print("CV score:", np.mean(scores))

sub.to_csv("submission_stacker.csv", index=False)
