__author__ = 'ando'
import os
import random
from multiprocessing import cpu_count
import logging as log


import numpy as np
import psutil
from math import floor
from ADSCModel.model import Model
from ADSCModel.context_embeddings import Context2Vec
from ADSCModel.node_embeddings import Node2Vec
from ADSCModel.community_embeddings import Community2Vec
import utils.IO_utils as io_utils
import utils.graph_utils as graph_utils
import utils.plot_utils as plot_utils
import timeit

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)




p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

if __name__ == "__main__":

    #Reading the input parameters form the configuration files
    number_walks = 10                       # number of walks for each node
    walk_length = 80                        # length of each walk
    representation_size = 2               # size of the embedding
    num_workers = 10                        # number of thread
    num_iter = 9                            # number of overall iteration
    reg_covar = 0.00001                          # regularization coefficient to ensure positive covar
    input_file = 'BlogCatalog'                          # name of the input file
    output_file = 'BlogCatalog'                         # name of the output file
    batch_size = 100
    window_size = 5    # windows size used to compute the context embedding
    negative = 5        # number of negative sample
    lr = 0.025            # learning rate

    """
    alpha = 1.0
    beta = 0.01
    num_iter_com = 1  # number of iteration for community embedding
    num_iter_node = 1  # number of iteration for node embedding
    """

    alpha_betas = (0.1, 1)

    weight_concentration_prior = 100
    walks_filebase = os.path.join('data', output_file, output_file)            # where read/write the sampled path
    sampling_path = False



    #CONSTRUCT THE GRAPH
    G = graph_utils.load_matfile(os.path.join('./data', input_file, input_file + '.mat'), undirected=True)
    # Sampling the random walks for context
    if sampling_path:
        log.info("sampling the paths")
        walk_files = graph_utils.write_walks_to_disk(G, walks_filebase + ".walks",
                                                     num_paths=number_walks,
                                                     path_length=walk_length,
                                                     alpha=0,
                                                     rand=random.Random(0),
                                                     num_workers=num_workers)
    else:
        walk_files = ["{}_n2v.walks.0{}".format(walks_filebase, i) for i in range(number_walks) if os.path.isfile("{}_n2v.walks.0{}".format(walks_filebase, i))]

    vertex_counts = graph_utils.count_textfiles(walk_files, num_workers)
    model = Model(vertex_counts,
                  size=representation_size,
                  table_size=100000000,
                  input_file=os.path.join(input_file, input_file),
                  path_labels="./data")


    #Learning algorithm
    node_learner = Node2Vec(workers=num_workers, negative=negative, lr=lr)
    cont_learner = Context2Vec(window_size=window_size, workers=num_workers, negative=negative, lr=lr)
    com_learner = Community2Vec(lr=lr)


    context_total_path = G.number_of_nodes() * number_walks * walk_length
    edges = np.array(G.edges())
    iter_node = 1
    iter_com = 1
    alpha, beta = alpha_betas

    log.info("_______________________")
    log.info("\t\tINITIAL LOSS\t\t")

    com_learner.fit(model, reg_covar=reg_covar, wc_prior=weight_concentration_prior, n_init=1)
    o3 = com_learner.loss(sorted(G.nodes()), model, beta)

    o1 = node_learner.loss(model, edges)
    o2 = cont_learner.loss(model, graph_utils.combine_files_iter(walk_files),
                      total_paths=context_total_path,
                      alpha=alpha)
    o3 = com_learner.loss(sorted(G.nodes()), model, beta)
    log.info("initial loss: {}\to1: {}\to2: {}\to3: {}".format(o1+o2+o3, o1, o2, o3))

    ##########################
    #   EMBEDDING LEARNING    #
    ###########################

    for it in range(num_iter):
        # for alpha, beta in alpha_betas:
        log.info('\n_______________________________________\n')
        log.info('\t\tITER-{}\n'.format(it))
        model = Model.load_model(path="data", file_name="BlogCatalog_alpha-0.1_beta-1_it-{}_ws-5_ng-5_rs-2".format(it))

        o1 = node_learner.loss(model, edges)
        o2 = cont_learner.loss(model, graph_utils.combine_files_iter(walk_files),
                               total_paths=context_total_path,
                               alpha=alpha)
        o3 = com_learner.loss(sorted(G.nodes()), model, beta)
        log.info("loss-{}: {}\to1: {}\to2: {}\to3: {}".format(it, o1 + o2 + o3, o1, o2, o3))




