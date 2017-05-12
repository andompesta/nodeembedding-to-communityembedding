__author__ = 'ando'
import sklearn.mixture as mixture
import numpy as np
import logging
from scipy.stats import multivariate_normal

logger = logging.getLogger("adsc")


class Community2Vec(object):
    '''
    Class that train the community embedding
    '''
    def __init__(self, reg_covar=0, is_debug=False, plot=False, save_predict_labels=False):
        '''
        :param alpha: learning rate
        :param window: windows size used to compute the context embeddings
        :param workers: number of thread
        :param min_alpha: min learning rate
        :param negative: number of negative samples
        :return:
        '''

        self.reg_covar = reg_covar
        self.is_debug = is_debug
        self.plot = plot
        self.save_predict_labels = save_predict_labels

    def loss(self, model, lambda2):
        ret_loss = 0
        for com in range(model.k):
            rd = multivariate_normal(model.centroid[com], model.covariance_mat[com])
            ret_loss += rd.pdf(model.node_embedding) * model.pi[:, com]
        ret_loss = sum(np.log(ret_loss))
        return ret_loss * (-lambda2/model.k)

    def train(self, model):
        '''
        Fit the GMM model with the current node embedding
        '''
        print("train community \t num community %d" % model.k)
        g = mixture.GaussianMixture(n_components=model.k, reg_covar=self.reg_covar, covariance_type='full', n_init=5)

        g.fit(model.node_embedding)

        # diag_covars = []
        # for covar in g.covariances_:
        #     diag = np.diag(covar)
        #     diag_covars.append(diag)

        model.centroid = np.float32(g.means_)
        model.covariance_mat = np.float32(g.covariances_)
        model.inv_covariance_mat = np.linalg.inv(model.covariance_mat)
        model.pi = np.float32(g.predict_proba(model.node_embedding))

        if self.save_predict_labels:
            model.predict_label = g.predict(model.node_embedding)

        if self.plot:
            model.predict_label = g.predict(model.node_embedding)
