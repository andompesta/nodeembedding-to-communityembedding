import configparser
import logging
import os
import random
import sys
from multiprocessing import cpu_count


import numpy as np
import psutil

from ADSCModel.model import Model
from ADSCModel.context_embeddings import Context2Vec
from ADSCModel.community_embeddings import Community2Vec

import utils.plot_utils as plot_utils
import utils.graph_utils as graph_utils


p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

#Setting the logger parameters
FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s\t|\t%(message)s"
level = logging.DEBUG
logging.basicConfig(format=FORMAT, level=level)
logger = logging.getLogger()

prop = configparser.ConfigParser()
prop.read('../conf.ini')




def process_context(context_learner, model, walks, _lambda1=1.0, _lambda2=0.1, total_nodes=None):
    logger.info("Training context...")
    context_learner.train(model=model, paths=walks, _lambda1=_lambda1, _lambda2=(_lambda2/(model.k * context_learner.window)), total_words=total_nodes)


if __name__ == "__main__":

    #Reading the input parameters form the configuration files
    number_walks = prop.getint('MY', 'number_walks')                      # number of walks for each node
    walk_length = prop.getint('MY', 'walk_length')                        # length of each walk
    window_size = prop.getint('MY', 'window_size')                        # windows size used to compute the context embedding
    negative = prop.getint('MY', 'negative')                              # number of negative sample
    representation_size = prop.getint('MY', 'representation_size')        # size of the embedding
    num_workers = prop.getint('MY', 'num_workers')                        # number of thread
    num_iter = prop.getint('MY', 'num_iter')                              # number of iteration
    reg_covar = prop.getfloat('MY', 'reg_covar')                          # regularization coefficient to ensure positive covar
    input_file = prop.get('MY', 'input_file_name')                          # name of the input file
    output_file = prop.get('MY', 'input_file_name')                         # name of the output file

    lambda_1_val = prop.getfloat('MY', 'lambda_1')                        # alpha parameter for O2
    lambda_2_val = prop.getfloat('MY', 'lambda_2')                        # beta parameter for O3
    down_sample = prop.getfloat('MY', 'down_sample')


    walks_filebase = '../data/' + output_file + ".walks"                  # where read/write the sampled path
    sampling_path = prop.getboolean('MY', 'sampling_path')                # execute sampling of new walks
    pretraining = prop.getboolean('MY', 'pretraining')                    # execute pretraining


    #CONSTRUCT THE GRAPH
    G = graph_utils.load_adjacencylist('../data/' + input_file + '/' + input_file + '.adjlist', True)
    node_color = plot_utils.graph_plot(G=G, save=False, show=False)

    # Sampling the random walks for context

    walk_files = None
    if sampling_path:
        logger.debug("sampling the paths")
        walk_files = graph_utils.write_walks_to_disk(G, walks_filebase,
                                                     num_paths=number_walks,
                                                     path_length=walk_length,
                                                     alpha=0,
                                                     rand=random.Random(9999999999),
                                                     num_workers=num_workers)
    else:
        walk_files = [walks_filebase + '.' + str(i) for i in range(number_walks)]

    vertex_counts = G.degree(G.nodes_iter())

    model = Model(G=G,
                  size=representation_size,
                  min_count=0,
                  table_size=1000000,
                  path_labels='../data/',
                  input_file=input_file + '/' + input_file,
                  vocabulary_counts=vertex_counts,
                  downsampling=down_sample)

    #Learning algorithm
    cont_learner = Context2Vec(window=window_size, workers=num_workers, negative=3)
    comm_learner = Community2Vec(reg_covar=reg_covar)

    context_total_path = G.number_of_nodes() * number_walks * walk_length
    logger.debug("context_total_node: %d" % (context_total_path))

    # Sampling edges for node embedding
    edges = np.array(G.edges())

    process_context(cont_learner, model, graph_utils.combine_files_iter(walk_files), _lambda1=lambda_1_val,
                    _lambda2=0, total_nodes=context_total_path)

    # plot_utils.node_space_plot_2D_elipsoid(model.node_embedding, node_color, grid=False)