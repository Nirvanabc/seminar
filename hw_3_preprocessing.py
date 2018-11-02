import re
import numpy as np
from hw_3_header import banks_set
### data preprocessing functions

def mean(x):
    '''
    x is a list of values,
    each value can be:
    '0', '1', '-1', 'None'
    we don't need None
    '''
    l = []
    for i in x:
        if i == '0': l.append(0)
        elif i == '1': l.append(1)
        elif i == '-1': l.append(-1)
    return int(round(np.mean(l)))


def prepare_text(text):
    '''
    leaves only lowercase letters
    '''
    text = text.lower()
    text = re.findall(r"[а-я]+", text)
    return ' '.join(text)


def prepare_data(root, label_set):
    '''
    returns X and y data
    '''
    curr_field = 0
    X = []
    y = []
    for child in root:
        for table in child:
            list_of_marks = []
            for column in table.findall('column'):
                name = column.get('name')
                if name in label_set:
                    # add marks of each bank
                    list_of_marks.append(column.text)
                    continue
                if name == 'text':
                     X.append(prepare_text(column.text))
            if list_of_marks != []:
                val = mean(list_of_marks)
                y.append(val)
    return np.array(X), np.array(y)

