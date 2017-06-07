__author__ = 'ando'

from os.path import join as path_join
import utils.graph_utils as graph_utils
import utils.IO_utils as io_utils
import utils.plot_utils as plot_utils
import sklearn.mixture as mixture
import numpy as np

input_file = 'karate'
node_embedding = io_utils.load_embedding(path='../data',
                                         file_name="{}_alpha-1.0_beta-0.1_ws-3_neg-4_ds-0_lr-0.1".format(input_file))

g = mixture.GaussianMixture(n_components=2, reg_covar=0.000001, covariance_type='full', n_init=5)
g.fit(node_embedding)
centroid = np.float32(g.means_)
covariance_mat = np.float32(g.covariances_)


G = graph_utils.load_adjacencylist(path_join("../data/", input_file, input_file + '.adjlist'), True)
node_color = plot_utils.graph_plot(G=G,
                                   show=False,
                                   graph_name="karate",
                                   node_position_file=True,
                                   node_position_path='../data')

plot_utils.node_space_plot_2D_elipsoid(node_embedding,
                                       means=centroid,
                                       covariances=covariance_mat,
                                       color_values=node_color,
                                       grid=False,
                                       show=True)
