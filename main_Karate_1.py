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
    number_walks = 10                      # number of walks for each node
    walk_length = 20                        # length of each walk
    representation_size = 2        # size of the embedding
    num_workers = 10                        # number of thread
    num_iter = 5                              # number of iteration
    reg_covar = 0.00001                          # regularization coefficient to ensure positive covar
    input_file = 'karate'                          # name of the input file
    output_file = 'karate'                         # name of the output file

    window_size = 3  # windows size used to compute the context embedding
    negative = 4  # number of negative sample
    alpha = 1.0
    beta = 0.01
    lr = 0.02
    weight_concentration_prior = 100
    walks_filebase = os.path.join('data', output_file, output_file + ".walks")            # where read/write the sampled path
    sampling_path = True



    #CONSTRUCT THE GRAPH
    G = graph_utils.load_adjacencylist(os.path.join('./data', input_file, input_file + '.adjlist'), True)
    node_color = plot_utils.graph_plot(G=G,
                                       show=False,
                                       graph_name="karate",
                                       node_position_file=True,
                                       node_position_path='./data')
    model = Model(G.degree(),
                  size=representation_size,
                  table_size=5000000,
                  input_file=os.path.join(input_file, input_file + "_zachary"),
                  path_labels="./data")

    model.node_color = node_color
    model.k = 2

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

    log.info('\n_______________________________________\n')
    log.info('using alpha 1:%.4f \t beta 2:%.4f' % (alpha, beta))
    log.debug('Number of community: %d' % model.k)

    ###########################
    #   PRE-TRAINING          #
    ###########################
    log.info("pre-train the model")
    node_learner.train(model,
                       edges=edges,
                       iter=1,
                       chunksize=20)

    cont_learner.train(model,
                       paths=graph_utils.combine_files_iter(walk_files),
                       total_nodes=context_total_path,
                       alpha=alpha,
                       chunksize=20)

    io_utils.save_embedding(model.node_embedding, "{}_pre-training".format(output_file))

    ###########################
    #   EMBEDDING LEARNING    #
    ###########################
    for it in range(1):
        log.info('\n_______________________________________\n')
        start_time = timeit.default_timer()

        node_learner.train(model,
                           edges=edges,
                           iter=1,
                           chunksize=20)

        cont_learner.train(model,
                           paths=graph_utils.combine_files_iter(walk_files),
                           total_nodes=context_total_path,
                           alpha=alpha,
                           chunksize=20)

        # com_learner.fit(model)
        # com_learner.train(G.nodes(), model, beta, chunksize=20, iter=7)
        # log.info('time: %.2fs' % (timeit.default_timer() - start_time))
        # log.info(model.centroid)

        # model.save("{}_alpha-{}_beta-{}_ws-{}_neg-{}_ds-{}_lr-{}_wc-{}".format(output_file,
        #                                                                        alpha,
        #                                                                        beta,
        #                                                                        window_size,
        #                                                                        negative,
        #                                                                        0,
        #                                                                        lr,
        #                                                                        weight_concentration_prior))

        # com_learner.fit(model)
        plot_utils.node_space_plot_2D_elipsoid(model.node_embedding,
                                               means=model.centroid,
                                               covariances=model.covariance_mat,
                                               color_values=node_color,
                                               grid=False,
                                               show=True)



