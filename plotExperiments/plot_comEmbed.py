__author__ = 'ando'

import utils.graph_utils as graph_utils
import utils.IO_utils as io_utils
import utils.plot_utils as plot_utils
from ADSCModel.model import Model
import sklearn.mixture as mixture
import numpy as np
import itertools


output_file = 'karate'
window_size = 5
negative = 5
representation_size = 2
down_sample = 0

lambda_1_val = 0.5
lambda_2_val = 0.001


for it in [0, 1, 2, 3, 4]:
    # print('l1:%.2f \t l2:%.2f \t it:%d'% (l1_val, l2_val, it+1))

    file_name = output_file + "_comEmb" + \
                "_l1-"+str(lambda_1_val) + \
                "_l2-"+str(lambda_2_val) + \
                "_ds-"+str(down_sample) + \
                "_it-"+str(it)
    print(file_name)
    model = Model.load_model(path='../data', file_name=file_name)

    color_iter = itertools.cycle(['red', 'cyan',  'purple','lightgreen'])
    plot_utils.node_space_plot_2D_elipsoid(model.node_embedding, model.node_color,
                                           means=model.centroid,
                                           covariances=model.covariance_mat,
                                           grid=True,
                                           color_iter=color_iter)

    # plot_utils.node_space_plot_2D(model['node_embedding'], save=False, color_values=model['ground_true'])
