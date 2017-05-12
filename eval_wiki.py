__author__ = 'ando'
from ADSCModel.model import Model
from sklearn.svm import LinearSVC
import configparser
import numpy as np
import utils.IO_utils as model_utils
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize
import logging
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import warnings

level = logging.DEBUG
logger = logging.getLogger('eval_adsc')
logger.setLevel(level)

prop = configparser.ConfigParser()
prop.read('conf.ini')


if __name__ == '__main__':

    # prefix = 'my_'
    # prefix = 'line_'
    # prefix = 'node2vec'
    # prefix = '_GraRep'
    prefix = '_autoencoder'

    seeds = {'BlogCatalog':[10, 78, 20, 30, 31, 74, 45, 50, 51, 79],
             'PPI': [13, 18, 22, 26, 29, 30, 34, 36, 43, 50],
             # 'Wikipedia': [14,8,16,73,79,96,98, 53, 43, 64]
             # 'Wikipedia': [6, 7, 47, 56, 59, 99, 83, 20, 13, 66]
             'Wikipedia': [7, 22, 31, 53, 72, 98, 79]
             }
    # seeds = range(100)



    C = [1]
    dwon_sampling = [0]
    iterations = [1]
    input_file = 'Wikipedia'

    X = None
    y = None
    for c in C:
        file_name = input_file + prefix
        print('%s\t%f' %(file_name, c))

        X = model_utils.load_embedding(path='data', file_name=file_name)
        y = model_utils.load_ground_true(path='data', file_name=input_file + '/' + input_file, multilabel=True)[0]

        X = normalize(X)
        lb = preprocessing.MultiLabelBinarizer()
        y = lb.fit_transform(y)

        # [print('node %d features %s' % (node_id, values)) for node_id, values in enumerate(X)]
        # [print('node %d label %d' % (node_id, values)) for node_id, values in enumerate(y)]
        clf = OneVsRestClassifier(LinearSVC(C=c, loss='squared_hinge', penalty='l2'))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for ratio in np.arange(0.1, 1, 0.1):
                avg_micro_f1 = []
                avg_macro_f1 = []
                for i, seed in enumerate(seeds[input_file]):
                # for i, seed in enumerate(seeds):

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=seed)
                    clf.fit(X_train, y_train)

                    y_pred = np.zeros(y_test.shape)
                    y_pred_prob = clf.decision_function(X_test)
                    for index in range(len(y_test)):
                        Y_test_index = np.where(y_test[index] == 1)[0]
                        row_pred = y_pred_prob[index].argpartition(-len(Y_test_index))[-len(Y_test_index):]
                        y_pred[index, row_pred] = 1

                    # y_pred = clf.predict(X_test)

                    micro_f1 = f1_score(y_test, y_pred, average='micro')
                    macro_f1 = f1_score(y_test, y_pred, average='macro')
                    avg_micro_f1.append(micro_f1)
                    avg_macro_f1.append(macro_f1)
                    print('%d\t%f\t%f\t%f' % (i, 1-ratio, micro_f1, macro_f1))

                print('%f\t%f\t%f' % (1-ratio, np.mean(avg_micro_f1), np.mean(avg_macro_f1)) )

