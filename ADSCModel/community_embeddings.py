__author__ = 'ando'
import sklearn.mixture as mixture
import numpy as np
import logging

logger = logging.getLogger("adsc")


def community_sdg(node_embedding, centroid, inv_covariance_mat, pi, k, _alpha, _lambda2, index):
    '''
      Perform stochastic gradient descent of the comunity embedding.
      NOTE: using the cython implementation (fast_community_sdg_X) is much more fast
    '''
    grad = np.zeros(node_embedding[index].shape, dtype=np.float32)
    if _lambda2 > 0:
        for com in range(k):
            diff = (node_embedding[index] - centroid[com])
            m = pi[index, com] * inv_covariance_mat[com]
            grad += np.dot(m, diff) * _lambda2
    return - np.clip((grad), -_alpha, _alpha)



def training(model, is_debug=False, plot=False):
    '''
    Fit the GMM model with the current node embedding
    '''
    logger.debug("num community %d" % model.k)
    g = mixture.GaussianMixture(n_components=model.k, covariance_type='diag', n_init=1)

    g.fit(model.node_embedding)

    diag_covars = []
    for covar in g.covariances_:
        diag = np.diag(covar)
        diag_covars.append(diag)

    model.centroid = np.float32(g.means_)
    model.inv_covariance_mat = np.float32(np.linalg.inv(diag_covars))
    model.pi = np.float32(g.predict_proba(model.node_embedding))

    if is_debug:
        print('mean')
        print(model.centroid)

        print('covar mtrix')
        print(model.inv_covariance_mat)

    if plot:
        model.predict_label = g.predict(model.node_embedding)