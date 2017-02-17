__author__ = 'ando'
from ADSCModel.model import Model
from sklearn.svm import LinearSVC
import configparser
import numpy as np
import utils.IO_utils as model_utils
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize
import logging
import sys
from sklearn.model_selection import train_test_split


level = logging.DEBUG
logger = logging.getLogger('eval_adsc')
logger.setLevel(level)

prop = configparser.ConfigParser()
prop.read('conf.ini')


if __name__ == '__main__':

    # prefix = '_comEmb_'
    # prefix = 'my_'
    # prefix = 'line_'
    # prefix = 'node2vec'
    prefix = '_GraRep'

    # seeds = range(100)


    input_file = 'Wikipedia'

    X = None
    y = None
    file_name = input_file + prefix
    print(file_name)

    X = model_utils.load_embedding_graphRep(path='data', file_name=file_name)['res']

    print(X.shape)

    model_utils.save_embedding(X, file_name=file_name, path='data')

