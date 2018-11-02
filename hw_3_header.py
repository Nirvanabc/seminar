import numpy as np
import xml.etree.ElementTree as ET
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from  hyperopt import hp, fmin, pyll, tpe
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

# for crossval
N_SPLITS = 5

# for fmin
MAX_EVALS = 3 # 150

# crossval
skf = StratifiedKFold(n_splits=N_SPLITS)

# tf-id vectorizer
vectorizer = TfidfVectorizer(max_features=None, binary=True, analyzer='char')

# possible values for hyperparamemers
ngram_range_list = [(2,3), (2,4), (2,5), (2,6), (3,4), (3,5), (3,6)]
n_neighbors_list = [1,2,3,4,5,6,7,8,9,10]
min_df_list = [1,2,3]
C_list = np.arange(0.1, 10.1, 0.3)

# labels
banks_set = ['sberbank',
             'vtb',
             'gazprom',
             'alfabank',
             'bankmoskvy',
             'raiffeisen',
             'uralsib',
             'rshb']

tcc_set = ['beeline',
           'mts',
           'megafon',
           'tele2',
           'rostelecom',
           'komstar',
           'skylink']

banks_train_file = 'SentiRuEval_2016/banks_train.xml'
banks_test_file = 'SentiRuEval_2016/banks_test.xml'

tcc_train_file = 'SentiRuEval_2016/tcc_train.xml'
tcc_test_file = 'SentiRuEval_2016/tcc_test.xml'
