__author__ = 'ando'

import utils.graph_utils as graph_utils
import utils.IO_utils as io_utils
import utils.plot_utils as plot_utils
import sklearn.mixture as mixture
import numpy as np


output_file = 'karate'
window_size = 5
negative = 5
representation_size = 2
down_sample = 0

lambda_1_vals = [0.1, 0.01, 0.001, 0.0001]
lambda_2_vals = [1, 0.1, 0.01, 0.001, 0.0001]
down_samples = [0]

for down_sample in down_samples:
    for l1_val in lambda_1_vals:
        for l2_val in lambda_2_vals:
            for it in [0,1,2,3,4]:
                print('l1:%.2f \t l2:%.2f \t it:%d'% (l1_val, l2_val, it+1))


                model = io_utils.load_model(path='../data', file_name=output_file + "_comEmb" +
                                                                      "_l1-"+str(l1_val) +
                                                                      "_l2-"+str(l2_val) +
                                                                      "_ds-"+str(down_sample) +
                                                                      "_iter-"+str(it))

                # plot_utils.node_space_plot_2D_elipsoid(model['node_embedding'], model['node_color'],
                #                                        labels=model['predict_label'], means=model['centroid'],
                #                                        covariances=model['covariance_mat'],
                #                                        grid=False)

                plot_utils.node_space_plot_2D_elipsoid(model['node_embedding'], model['node_color'],
                                                       labels=model['predict_label'], means=None,
                                                       covariances=None,
                                                       grid=False,
                                                       path='/Users/ando/Desktop/IJCL_plot/Embedding/rest/karate'+
                                                            '_l1-'+str(l1_val) +
                                                            '_l2-'+str(l2_val) +
                                                            '_it-'+str(it))
