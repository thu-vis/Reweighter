# import pickle
import _pickle as pickle
import numpy as np
import os
import threading
import sys
import json as json
from sklearn.metrics import confusion_matrix, roc_auc_score, \
    precision_recall_curve, auc, roc_curve
from sklearn.preprocessing import OneHotEncoder

from threading import Thread
from time import sleep

exec_list = {}

# Pickle loading and saving
def pickle_save_data(filename, data):
    try:
        pickle.dump(data, open(filename, "wb"))
    except Exception as e:
        print(e, end=" ")
        print("So we use the highest protocol.")
        pickle.dump(data, open(filename, "wb"), protocol=4)
    return True


def pickle_load_data(filename):
    try:
        mat = pickle.load(open(filename, "rb"))
    except Exception:
        mat = pickle.load(open(filename, "rb"))
    return mat


# json loading and saving
def json_save_data(filename, data):
    with open(filename, "w") as f:
        f.write(json.dumps(data))
    return True


def json_load_data(filename, encoding=None):
    with open(filename, "r", encoding=encoding) as f:
        return json.load(f)


# directory
def check_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return True


def check_exist(filename):
    return os.path.exists(filename)


# normalization
def unit_norm_for_each_col(X):
    X -= X.min(axis=0)
    X /= X.max(axis=0)
    return X


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def one_hot_encoder(List):
    res = np.zeros((len(List), max(List) + 1))
    res[np.array(range(len(List))), List] = 1
    return res



# metrics
def accuracy(y_true, y_pred, weights=None):
    score = (y_true == y_pred)
    return np.average(score, weights=weights)

