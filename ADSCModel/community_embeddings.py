__author__ = 'ando'
import sklearn.mixture as mixture
import numpy as np
from utils.embedding import chunkize_serial, RepeatCorpusNTimes

from scipy.stats import multivariate_normal
import logging as log

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)


class Community2Vec(object):
    '''
    Class that train the community embedding
    '''
    def __init__(self, model, lr, reg_covar=0):
        self.lr = lr
        self.g_mixture = mixture.GaussianMixture(n_components=model.k, reg_covar=reg_covar, covariance_type='full', n_init=10)

    def fit(self, model):
        '''
        Fit the GMM model with the current node embedding and save the result in the model
        :param model: model injected to add the mixture parameters
        '''

        log.info("Fitting: {} communities".format(model.k))
        self.g_mixture.fit(model.node_embedding)

        # diag_covars = []
        # for covar in g.covariances_:
        #     diag = np.diag(covar)
        #     diag_covars.append(diag)

        model.centroid = self.g_mixture.means_.astype(np.float32)
        model.covariance_mat = self.g_mixture.covariances_.astype(np.float32)
        model.inv_covariance_mat = np.linalg.inv(model.covariance_mat).astype(np.float32)
        model.pi = self.g_mixture.predict_proba(model.node_embedding).astype(np.float32)


    def loss(self, nodes, model, beta, chunksize=150):
        """
        Forward function used to compute o3 loss
        :param input_labels: of the node present in the batch
        :param model: model containing all the shared data
        :param beta: trade off param
        """
        ret_loss = 0
        for node_index in chunkize_serial(map(lambda x: model.vocab(x).index, nodes), chunksize):
            input = model.node_embedding[node_index]

            batch_loss = np.zeros(len(node_index), dtype=np.float32)
            for com in range(model.k):
                rd = multivariate_normal(model.centroid[com], model.covariance_mat[com])
                # check if can be done as matrix operation
                batch_loss += rd.logpdf(input).astype(np.float32) * model.pi[node_index, com]

            ret_loss = abs(batch_loss.sum())

        return ret_loss * (beta/model.k)

    def train(self, nodes, model, beta, chunksize=150, iter=1):
        for _ in range(iter):
            grad_input = np.zeros(model.node_embedding.shape).astype(np.float32)
            for node_index in chunkize_serial(map(lambda x: model.vocab[x].index, nodes), chunksize):
                # log.debug(input)
                # log.debug(input.cpu().numpy() - self.centroid[0])
                # log.debug(grad_output)
                input = model.node_embedding[node_index]
                batch_grad_input = np.zeros(input.shape).astype(np.float32)

                for com in range(model.k):
                    diff = np.expand_dims(input - model.centroid[com], axis=-1)
                    m = model.pi[node_index, com].reshape(len(node_index), 1, 1) * model.inv_covariance_mat[com]

                    batch_grad_input += np.squeeze(np.matmul(m, diff), axis=-1)
                grad_input[node_index] += batch_grad_input
                # log.debug("m: {}".format(m))
                # log.debug("grad: {}".format(grad_input))

            grad_input *= (beta/model.k)

            model.node_embedding -= (grad_input.clip(min=-5, max=5)) * self.lr



    #
    # def loss(self, model, lambda2):
    #     ret_loss = 0
    #     for com in range(model.k):
    #         rd = multivariate_normal(model.centroid[com], model.covariance_mat[com])
    #         ret_loss += rd.pdf(model.node_embedding) * model.pi[:, com]
    #     ret_loss = sum(np.log(ret_loss))
    #     return ret_loss * (-lambda2/model.k)
    #
    # def fit(self, model):
    #     '''
    #     Fit the GMM model with the current node embedding
    #     '''
    #     print("train community \t num community %d" % model.k)
    #     g = mixture.GaussianMixture(n_components=model.k, reg_covar=self.reg_covar, covariance_type='full', n_init=5)
    #
    #     g.fit(model.node_embedding)
    #
    #     # diag_covars = []
    #     # for covar in g.covariances_:
    #     #     diag = np.diag(covar)
    #     #     diag_covars.append(diag)
    #
    #     model.centroid = np.float32(g.means_)
    #     model.covariance_mat = np.float32(g.covariances_)
    #     model.inv_covariance_mat = np.linalg.inv(model.covariance_mat)
    #     model.pi = np.float32(g.predict_proba(model.node_embedding))
    #
    #     if self.save_predict_labels:
    #         model.predict_label = g.predict(model.node_embedding)
    #
    #     if self.plot:
    #         model.predict_label = g.predict(model.node_embedding)
