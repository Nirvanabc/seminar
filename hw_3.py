from hw_3_header import *
from hw_3_preprocessing import *

# how to pass X,y as a parameter to objective_knn?

### evaluating functions
# knn
def evaluate_knn(X, y, X_test, y_test, ngram_range, n_neighbors, min_df):
    model = pipe_knn.set_params(vectorizer__ngram_range=ngram_range,
                                vectorizer__min_df=min_df,
                                knn__n_neighbors=n_neighbors).fit(X, y)
    y_pred = pipe_knn.predict(X_test)
    # we use '-' as fmin searches for minimum, but f1 should be maximized
    return -f1_score(y_test, y_pred, average='macro')


def objective_knn(args):
    ngram_range = args['ngram_range']
    n_neighbors = args ['n_neighbors']
    min_df = args['min_df']
    pred = np.zeros(N_SPLITS)
    for i, [train_index, test_index] in enumerate(skf.split(X,y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pred[i] = evaluate_knn(X_train,
                           y_train,
                           X_test,
                           y_test,
                           ngram_range,
                           n_neighbors,
                           min_df)
    predicted = pred.mean()
    # print(predicted, ngram_range, n_neighbors, min_df)
    return pred.mean()

# svc
def evaluate_svc(X, y, X_test, y_test, C, ngram_range, min_df):
    model = pipe_svc.set_params(vectorizer__ngram_range=ngram_range,
                            vectorizer__min_df=min_df,
                            svc__C=C).fit(X,y)
    y_pred = pipe_svc.predict(X_test)
    return -f1_score(y_test, y_pred, average='macro')
    

def objective_svc(args):
    C = args['C']
    ngram_range = args['ngram_range']
    min_df = args['min_df']
    pred = np.zeros(N_SPLITS)
    for i, [train_index, test_index] in enumerate(skf.split(X,y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pred[i] = evaluate_svc(X_train,
                               y_train,
                               X_test,
                               y_test,
                               C,
                               ngram_range,
                               min_df)
    predicted = pred.mean()
    print(predicted, pred, C, ngram_range, min_df)
    return pred.mean()


def best_hyperparam():
    # fmin returns the position of the best value in respective list
    best_knn = fmin(
        fn=objective_knn,
        space=space_knn,
        algo=tpe.suggest,
        max_evals=MAX_EVALS)

    best_svc = fmin(
        fn=objective_svc,
        space=space_svc,
        algo=tpe.suggest,
        max_evals=MAX_EVALS)

    return best_knn, best_svc


def f1_score_on_test():
    f1_score_knn = -evaluate_knn(
        X,
        y,
        X_test,
        y_test,
        ngram_range_list[best_knn_banks['ngram_range']],
        n_neighbors_list[best_knn_banks['n_neighbors']],
        min_df_list[best_knn_banks['min_df']])
    
    f1_score_svc = -evaluate_svc(
        X,
        y,
        X_test,
        y_test,
        C_list[best_svc_banks['C']],
        ngram_range_list[best_svc_banks['ngram_range']],
        min_df_list[best_svc_banks['min_df']])

    return f1_score_knn, f1_score_svc


### tuning hyperparameters block
## prepairings
# prepairings for pipeline with knn
clf_knn = KNeighborsClassifier() # metric='jaccard')
pipe_knn = Pipeline([('vectorizer', vectorizer), ('knn', clf_knn)])

space_knn = {
    'ngram_range': hp.choice('ngram_range', ngram_range_list),
    'n_neighbors': hp.choice('n_neighbors', n_neighbors_list),
    'min_df': hp.choice('min_df', min_df_list)
}

# prepairings for pipeline with LinearSVC
clf_svc = LinearSVC(class_weight='balanced')
pipe_svc = Pipeline([('vectorizer', vectorizer), ('svc', clf_svc)])
space_svc = {
    'C': hp.choice('C', C_list),
    'ngram_range': hp.choice('ngram_range', ngram_range_list),
    'min_df': hp.choice('min_df', min_df_list)
}

### data preprocessing
## banks
# train data
tree = ET.parse(banks_train_file)
root = tree.getroot()
X, y = prepare_data(root, banks_set)

# test data
tree_test = ET.parse(banks_test_file)
root_test = tree_test.getroot()
X_test, y_test = prepare_data(root_test, banks_set)

# best_knn_banks, best_svc_banks = best_hyperparam()
# f1_score_knn_banks, f1_score_svc_banks = f1_score_on_test()
# 
# ## tcc
# # train data
# tree = ET.parse(tcc_train_file)
# root = tree.getroot()
# X, y = prepare_data(root, tcc_set)
# 
# # test data
# tree_test = ET.parse(tcc_test_file)
# root_test = tree_test.getroot()
# X_test, y_test = prepare_data(root_test, tcc_set)
# 
# best_knn_tcc, best_svc_tcc = best_hyperparam()
# f1_score_knn_tcc, f1_score_svc_tcc = f1_score_on_test()
# 
# 
# print('f1_score_knn_banks {:.4}, f1_score_svc_banks {:.4}\n'.format(
#     f1_score_knn_banks, f1_score_svc_banks))
#   
# print('f1_score_knn_tcc {:.4}, f1_score_svc_tcc {:.4}\n'.format(
#     f1_score_knn_tcc, f1_score_svc_tcc))
# 
