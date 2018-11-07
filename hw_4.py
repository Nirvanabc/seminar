from hw_4_header import *
import pandas as pd
import hw_4_preprocessing
from hyperopt import hp, fmin, pyll, tpe
from sklearn.metrics import f1_score
from hw_4_log_reg import *
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

### evaluating functions
def evaluate(X, y, X_test, y_test, lr, lambda_, b):
    model = Logistic_Regression(lr=lr,
                                num_iter=NUM_ITER,
                                lambda_=lambda_,
                                b=b)
    model.fit_SGD(X, y)
    y_pred = model.predict(X_test)
    
    ## I use this to make sure that there are not only zeros in my
    ## predictions.
    # print(np.count_nonzero(y_pred), len(y_pred))

    # we use '-' as fmin searches for minimum, but f1 should be maximized
    return -roc_auc_score(y_test, y_pred)


def objective(args):
    lr = args['lr']
    lambda_ = args ['lambda_']
    b = args['b']
    pred = np.zeros(N_SPLITS)
    for i, [train_index, test_index] in enumerate(skf.split(X,y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pred[i] = evaluate(X_train,
                           y_train,
                           X_test,
                           y_test,
                           lr,
                           lambda_,
                           b)
    predicted = pred.mean()

    # look at intermediate results
    print("pred {:.4f}, lr {:.2f}, lambda {:.5f}, b {:.2f} ".format(
        -predicted,
        lr,
        lambda_,
        b))
    return predicted


def best_hyperparam():
    # fmin returns the position of the best value in respective list
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=MAX_EVALS)
    return best


def score_on_test(best_list, X, y, X_test, y_test):
    score = -evaluate(
        X,
        y,
        X_test,
        y_test,
        lr_list[best_list['lr']],
        lambda_list[best_list['lambda_']],
        b_list[best_list['b']])
    return score


### tuning hyperparameters block

space = {
    'lr': hp.choice('lr', lr_list),
    'lambda_': hp.choice('lambda_', lambda_list),
    'b': hp.choice('b', b_list)
}

### data preprocessing
X = hw_4_preprocessing.prepare_train_X()
X_test = hw_4_preprocessing.prepare_test_X()
# X_test = hw_4_preprocessing.prepare_train_X()

scores = []
for class_name in CLASSES:
    # if you don't transfer it to array, you will have a mistake in fit as
    # lines will be counted starting not from zero
    y = np.array(hw_4_preprocessing.train[class_name])
    y_test = np.array(hw_4_preprocessing.test_y[class_name])    

    best_list = best_hyperparam()
    score = score_on_test(best_list, X, y, X_test, y_test)
    scores.append(score)
    print('score for class {} is {:.4f}\n'.format(class_name, score))

print('score {:.4}\n'.format(np.mean(scores)))
