__author__ = 'ando'
from sklearn.svm import LinearSVC
import configparser
import numpy as np
import utils.IO_utils as model_utils
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split




if __name__ == '__main__':

    # prefix = '_comEmb_pipeline'
    # prefix = 'line_'
    # prefix = 'node2vec'
    # prefix = '_deepWalk'
    prefix = '_autoencode'

    seeds = {'BlogCatalog': [10, 78, 20, 30, 31, 74, 45, 50, 51, 79],
             'PPI': [13, 18, 22, 26, 29, 30, 34, 36, 43, 50],
             # 'Wikipedia': [14,8,16,73,79,96,98, 53, 43, 64]
             # 'Wikipedia': [6, 7, 47, 56, 59, 99, 83, 20, 13, 66]
             'Wikipedia': [7, 22, 31, 53, 72, 98, 79],
             'Flickr' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
             }
    C = [1]

    input_file = 'Wikipedia'

    X = None
    y = None
    for c in C:
        print('\n_______________________________________\n')
        file_name = input_file + prefix
        print(file_name)

        X = model_utils.load_embedding(path='data', file_name=file_name)
        y = model_utils.load_ground_true(path='data', file_name=input_file + '/' + input_file)[0]

        # X = normalize(X)
        # [print('node %d features %s' % (node_id, values)) for node_id, values in enumerate(X)]
        # [print('node %d label %d' % (node_id, values)) for node_id, values in enumerate(y)]

        for ratio in np.arange(0.9, 1, 0.01):
            avg_micro_f1 = []
            avg_macro_f1 = []
            for i, seed in enumerate(seeds[input_file]):

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=seed)
                clf = LinearSVC(C=c)
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                micro_f1 = f1_score(y_test, y_pred, average='micro')
                macro_f1 = f1_score(y_test, y_pred, average='macro')
                avg_micro_f1.append(micro_f1)
                avg_macro_f1.append(macro_f1)
                print('%d\t%f\t%f\t%f' % (i, 1-ratio, micro_f1, macro_f1))

            print('%f\t%f\t%f' % (1-ratio, np.mean(avg_micro_f1), np.mean(avg_macro_f1)) )

