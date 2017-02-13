__author__ = 'ando'
import configparser
import os
import random
import sys
from multiprocessing import cpu_count


import numpy as np
import psutil

from ADSCModel.model import Model
from ADSCModel.context_embeddings import Context2Vec
from ADSCModel.node_embeddings import Node2Vec
from ADSCModel.community_embeddings import Community2Vec

import utils.IO_utils as io_utils
import utils.graph_utils as graph_utils
import utils.plot_utils as plot_utils
import logging




p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

#Setting the logger parameters
FORMAT = "%(levelname)s %(filename)s: %(lineno)s\t|\t%(message)s"
level = logging.DEBUG
logger = logging.getLogger('adsc')
logger.setLevel(level)


prop = configparser.ConfigParser()
prop.read('conf.ini')



def process_context(context_learner, model, walks, _lambda1=1.0, _lambda2=0.1, total_nodes=None):
    logger.info("Training context...")
    return context_learner.train(model=model, paths=walks, _lambda1=_lambda1, _lambda2=(_lambda2/(model.k * cont_learner.window_size)), total_words=total_nodes)


def process_node(node_learner, model, edges, iter=1, lambda2=0.0):
    logger.info("Training node embedding...")
    return node_learner.train(model, edges=edges, iter=iter, _lambda2=(lambda2/model.k))

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

    # lambda_1_val = float(prop.get('MY', 'lambda_1'))                        # alpha parameter for O2
    # lambda_2_val = float(prop.get('MY', 'lambda_2'))                        # beta parameter for O3
    # down_sample = float(prop.get('MY', 'down_sample'))


    walks_filebase = 'data/' + output_file + ".walks"                       # where read/write the sampled path
    sampling_path = prop.getboolean('MY', 'sampling_path')                  # execute sampling of new walks
    pretraining = prop.getboolean('MY', 'pretraining')                      # execute pretraining



    #CONSTRUCT THE GRAPH
    G = graph_utils.load_adjacencylist('data/' + input_file + '/' + input_file + '.adjlist', True)
    # node_color = plot_utils.graph_plot(G=G, save=False, show=False)

    # Sampling the random walks for context
    walk_files = [walks_filebase + '.' + str(i) for i in range(number_walks)]



    #Learning algorithm
    node_learner = Node2Vec(workers=num_workers, negative=negative)
    cont_learner = Context2Vec(window_size=window_size, workers=num_workers, negative=negative)
    comm_learner = Community2Vec(reg_covar=reg_covar)


    vertex_counts = G.degree(G.nodes_iter())

    model = Model(G=G,
                  size=representation_size,
                  min_count=0,
                  table_size=5000000,
                  input_file=input_file + '/' + input_file,
                  vocabulary_counts=vertex_counts,
                  downsampling=0)


    context_total_path = G.number_of_nodes() * number_walks * walk_length
    logger.info("context_total_node: %d" % (context_total_path))
    edges = np.array(G.edges())


    logger.info('\n_______________________________________\n')
    model = model.load_model(path='data', file_name=output_file+'_comEmb')

    logger.info('Number of community: %d' % model.k)
    logger.debug('Number of edges: %d' % len(edges)*2)

    ###########################
    #   EMBEDDING LEARNING    #
    ###########################
    for it in range(1):
        logging.info('\n_______________________________________\n')
        comm_learner.train(model)
        node_learner.train(model, edges=edges, iter=1, _lambda2=1.0, _lambda1=0)
        io_utils.save_embedding(model.node_embedding, file_name=output_file + "_comEmb_pipeline")
        model.save(path='data', file_name=output_file + "_comEmb_pipeline")
