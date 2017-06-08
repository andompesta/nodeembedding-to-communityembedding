__author__ = 'ando'
import os
import random
from multiprocessing import cpu_count
import logging as log


import numpy as np
import psutil

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
    representation_size = 128               # size of the embedding
    num_workers = 10                        # number of thread
    num_iter = 1                            # number of overall iteration
    reg_covar = 0.00001                          # regularization coefficient to ensure positive covar
    input_file = 'BlogCatalog'                          # name of the input file
    output_file = 'BlogCatalog'                         # name of the output file
    batch_size = 100
    window_size = 10    # windows size used to compute the context embedding
    negative = 5        # number of negative sample
    lr = 0.1            # learning rate

    """
    alpha = 1.0
    beta = 0.01
    num_iter_com = 1  # number of iteration for community embedding
    num_iter_node = 1  # number of iteration for node embedding
    """

    alpha_betas = [(1.0, 0.1), (0.01, 0.1), (0.001, 0.1),
                   (0.1, 1.0), (0.1, 0.01), (0.1, 0.001),
                   (0.1, 0.1)]
    iter_com_node = [(5, 1), (10, 1), (50, 1),
                     (1, 5), (1, 10), (1, 50),
                     (1, 1)]

    weight_concentration_prior = 100
    walks_filebase = os.path.join('data', output_file, output_file + ".walks")            # where read/write the sampled path
    sampling_path = True



    #CONSTRUCT THE GRAPH
    G = graph_utils.load_adjacencylist(os.path.join('./data', input_file, input_file + '.adjlist'), True)
    model = Model(G.degree(),
                  size=representation_size,
                  table_size=8000000,
                  input_file=os.path.join(input_file, input_file),
                  path_labels="./data")

    # Sampling the random walks for context
    if sampling_path:
        log.info("sampling the paths")
        walk_files = graph_utils.write_walks_to_disk(G, walks_filebase,
                                                     num_paths=number_walks,
                                                     path_length=walk_length,
                                                     alpha=0,
                                                     rand=random.Random(9999999999),
                                                     num_workers=num_workers)
    else:
        walk_files = [walks_filebase + '.' + str(i) for i in range(number_walks) if os.path.isfile(walks_filebase + '.' + str(i))]



    #Learning algorithm
    node_learner = Node2Vec(workers=num_workers, negative=negative, lr=lr)
    cont_learner = Context2Vec(window_size=window_size, workers=num_workers, negative=negative, lr=lr)
    com_learner = Community2Vec(model, reg_covar=reg_covar, lr=lr, wc_prior=weight_concentration_prior)


    context_total_path = G.number_of_nodes() * number_walks * walk_length
    edges = np.array(G.edges())
    log.debug("context_total_path: %d" % (context_total_path))
    log.debug('node total edges: %d' % G.number_of_edges())

    log.info('\n_______________________________________')
    log.info('\t\tPRE-TRAINING\n')
    ###########################
    #   PRE-TRAINING          #
    ###########################
    node_learner.train(model,
                       edges=edges,
                       iter=1,
                       chunksize=batch_size)

    cont_learner.train(model,
                       paths=graph_utils.combine_files_iter(walk_files),
                       total_nodes=context_total_path,
                       alpha=1.0,
                       chunksize=batch_size)

    io_utils.save_embedding(model.node_embedding, "{}_pre-training".format(output_file))

    ###########################
    #   EMBEDDING LEARNING    #
    ###########################
    for it in range(num_iter):
        for iter_com, iter_node in iter_com_node:
            for alpha, beta in alpha_betas:
                log.info('\n_______________________________________\n')
                log.info('\t\tITER-{}\n'.format(it))

                log.info('using alpha:{} \t beta:{} \t iter_com:{} \t iter_node: {}'.format(alpha, beta, iter_com, iter_node))
                log.debug('Number of community: %d' % model.k)

                start_time = timeit.default_timer()

                node_learner.train(model,
                                   edges=edges,
                                   iter=iter_node,
                                   chunksize=batch_size)

                cont_learner.train(model,
                                   paths=graph_utils.combine_files_iter(walk_files),
                                   total_nodes=context_total_path,
                                   alpha=alpha,
                                   chunksize=batch_size)

                com_learner.fit(model)
                com_learner.train(G.nodes(), model, beta, chunksize=batch_size, iter=iter_com)
                log.info('time: %.2fs' % (timeit.default_timer() - start_time))
                log.info(model.centroid)
                io_utils.save("{}_alpha-{0:0>4}_beta-{0:0>4}_ws-{}_neg-{}_lr-{}_wc-{}_icom-{}_ind-{}"
                              .format(output_file,
                                      alpha,
                                      beta,
                                      window_size,
                                      negative,
                                      lr,
                                      weight_concentration_prior,
                                      iter_com,
                                      iter_node))

