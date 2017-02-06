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

    prefix = '_comEmb_'
    # prefix = 'my_'
    # prefix = 'line_'
    # prefix = 'node2vec'
    seeds = [8, 45, 49, 51, 30, 42, 1, 3, 35, 24]

    # seeds = range(100)


    values = [ (0.1, 0.001), (1, 0.1), (0.1, 0.01), (0.1, 1),
               (0.001, 0.1), (0.01, 0.1), (0.1, 0.1)]
    C = [1]
    dwon_sampling = [0.001]
    input_file = 'BlogCatalog'

    X = None
    y = None
    for c in C:
        for ds in dwon_sampling:
            for lambda_1_val, lambda_2_val in values:
                for it in [1]:
                    logger.info('\n_______________________________________\n')
                    logger.info('using c:%.2f \t l1:%.4f \tl2:%.4f ' % (c, lambda_1_val, lambda_2_val))

                    file_name = input_file + prefix + \
                                'l1-' + str(lambda_1_val) + \
                                '_l2-' + str(lambda_2_val) + \
                                '_ds-' + str(ds) + \
                                '_it-' +str(it)
                    logger.info(file_name)

                    X = model_utils.load_embedding(path='data', file_name=file_name)
                    y = model_utils.load_ground_true(path='data', file_name=input_file + '/' + input_file)[0]

                    # X = normalize(X)
                    # [print('node %d features %s' % (node_id, values)) for node_id, values in enumerate(X)]
                    # [print('node %d label %d' % (node_id, values)) for node_id, values in enumerate(y)]

                    for ratio in np.arange(0.1, 0.3, 0.1):
                        avg_micro_f1 = []
                        avg_macro_f1 = []
                        for i, seed in enumerate(seeds):

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=seed)
                            clf = LinearSVC(C=c)
                            clf.fit(X_train, y_train)

                            y_pred = clf.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            micro_f1 = f1_score(y_test, y_pred, average='micro')
                            macro_f1 = f1_score(y_test, y_pred, average='macro')
                            avg_micro_f1.append(micro_f1)
                            avg_macro_f1.append(macro_f1)
                            logger.info('%d\t%f\t%f\t%f' % (i, 1-ratio, micro_f1, macro_f1))

                        logger.info('%f\t%f\t%f' % (1-ratio, np.mean(avg_micro_f1), np.mean(avg_macro_f1)) )

