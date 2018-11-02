from hw_4_header import *
from hw_4_preprocessing import *
from sklearn.model_selection import StratifiedKFold
from  hyperopt import hp, fmin, pyll, tpe
# import numpy as np

### evaluating functions
def evaluate(X, y, X_test, y_test, lr, lambda_, b):
    model = Logistic_Regression(lr=lr,
                                num_iter=NUM_ITER,
                                lambda_=lambda_,
                                b=b)
    model.fit_SGD(X, y)
    y_pred = model.predict(X_test)
    return -np.mean(y_pred == y_test)
#     # we use '-' as fmin searches for minimum, but f1 should be maximized
#     return -f1_score(y_test, y_pred, average='macro')


def objective(args):
    lr = args['lr']
    lambda_ = args ['lambda_']
    b = args['b']
    pred = np.zeros(N_SPLITS)
    predicted = evaluate(X,
                         y,
                         X_test,
                         y_test,
                         lr,
                         lambda_,
                         b)
    # for i, [train_index, test_index] in enumerate(skf.split(X,y)):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     pred[i] = evaluate(X_train,
    #                        y_train,
    #                        X_test,
    #                        y_test,
    #                        lr,
    #                        lambda_,
    #                        b)
    # predicted = pred.mean()
    print(predicted, lr, lambda_, b)
    return pred.mean()


def best_hyperparam():
    # fmin returns the position of the best value in respective list
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=MAX_EVALS)

    return best


def score_on_test():
    score = -evaluate(
        X,
        y,
        X_test,
        y_test,
        lr_list[best_list['lr']],
        lambda_list[best_list['lambda_']],
        b_list[best_list['b']]
        )
    return score


### tuning hyperparameters block
## prepairings
# prepairings for pipeline with knn

space = {
    'lr': hp.choice('lr', lr_list),
    'lambda_': hp.choice('lambda_', lambda_list),
    'b': hp.choice('b', b_list)
}


### data preprocessing
# train data
X, y = prepare_data()

# test data
X_test, y_test = prepare_data()

best_list = best_hyperparam()
score = score_on_test()

print('score {:.4}\n'.format(score))
