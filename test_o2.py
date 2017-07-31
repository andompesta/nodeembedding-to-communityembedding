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
import utils.IO_utils as io_utils
import utils.graph_utils as graph_utils
import timeit

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)


if __name__ == "__main__":

    #Reading the input parameters form the configuration files
    number_walks = 10                       # number of walks for each node
    walk_length = 80                        # length of each walk
    representation_size = 128               # size of the embedding
    num_workers = 10                        # number of thread
    num_iter = 1                            # number of overall iteration
    reg_covar = 0.00001                     # regularization coefficient to ensure positive covar
    input_file = 'Rochester'                # name of the input file
    output_file = 'Rochester'               # name of the output file
    batch_size = 100
    window_size = 10    # windows size used to compute the context embedding
    negative = 5        # number of negative sample
    lr = 0.025            # learning rate

    """
    alpha = 1.0
    beta = 0.01
    num_iter_com = 1  # number of iteration for community embedding
    num_iter_node = 1  # number of iteration for node embedding
    """

    down_sampling = 0.001
    p = 0.25
    q = 0.25
    walks_filebase = os.path.join('data', output_file)            # where read/write the sampled path
    sampling_path = False



    #CONSTRUCT THE GRAPH
    G = graph_utils.load_matfile(os.path.join('./data', input_file, input_file + '.mat'), undirected=True)
    # Sampling the random walks for context
    if sampling_path:
        log.info("sampling the paths")
        walk_files = graph_utils.write_walks_to_disk(G, os.path.join(walks_filebase, "{}.walks".format(output_file)),
                                                     num_paths=number_walks,
                                                     path_length=walk_length,
                                                     alpha=0,
                                                     rand=random.Random(0),
                                                     num_workers=num_workers)
    else:
        walks_filebase = os.path.join(walks_filebase, "n2v_path_p-{}_q-{}".format(p, q), output_file)
        walk_files = ["{}_n2v.walks.0{}".format(walks_filebase, i) for i in range(number_walks) if os.path.isfile("{}_n2v.walks.0{}".format(walks_filebase, i))]

    vertex_counts = graph_utils.count_textfiles(walk_files, num_workers)
    model = Model(vertex_counts,
                  size=representation_size,
                  down_sampling=down_sampling,
                  table_size=5000000,
                  input_file=os.path.join(input_file, input_file),
                  path_labels="./data")

    cont_learner = Context2Vec(window_size=window_size, workers=num_workers, negative=negative, lr=lr)

    context_total_path = G.number_of_nodes() * number_walks * walk_length
    log.debug("context_total_path: %d" % (context_total_path))
    log.debug('node total edges: %d' % G.number_of_edges())

    log.info('\n_______________________________________')
    log.info('\t\tPRE-TRAINING\n')
    ###########################
    #   PRE-TRAINING          #
    ###########################
    cont_learner.train(model,
                       paths=graph_utils.combine_files_iter(walk_files),
                       total_nodes=context_total_path,
                       alpha=1.0,
                       chunksize=batch_size)

    io_utils.save_embedding(model.node_embedding, model.vocab,
                            path="/home/ando/Project/nodeembeddingeval/data/node2vec",
                            file_name="{}_node2vec_my_p-{}_q-{}".format(output_file,
                                                                        p,
                                                                        q))

