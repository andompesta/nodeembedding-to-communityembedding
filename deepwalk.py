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
    walk_length = 80                        # length of each walk
    representation_size = 128        # size of the embedding
    num_workers = 10                        # number of thread
    batch_size = 100
    input_file = 'BlogCatalog'                          # name of the input file
    output_file = 'BlogCatalog_my_deepwalk'                         # name of the output file

    window_size = 10  # windows size used to compute the context embedding
    negative = 5  # number of negative sample
    lr = 0.025
    walks_filebase = os.path.join('data', input_file, input_file + ".walks")            # where read/write the sampled path
    sampling_path = True



    #CONSTRUCT THE GRAPH
    G = graph_utils.load_matfile(os.path.join('./data', input_file, input_file + '.mat'), undirected=True)
    model = Model(G.degree(),
                  size=representation_size,
                  input_file=os.path.join(input_file, input_file),
                  path_labels="./data",
                  table_size=100000000)

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
    cont_learner = Context2Vec(window_size=window_size, workers=num_workers, negative=negative, lr=lr)


    context_total_path = G.number_of_nodes() * number_walks * walk_length
    log.info("context_total_path: %d" % (context_total_path))
    log.info('node total edges: %d' % G.number_of_nodes())
    log.info('node total edges: %d' % G.number_of_edges())

    log.info('\n_______________________________________\n')
    ###########################
    #   EMBEDDING LEARNING    #
    ###########################
    for it in range(1):
        log.info('\n_______________________________________\n')
        start_time = timeit.default_timer()
        cont_learner.train(model,
                           paths=graph_utils.combine_files_iter(walk_files),
                           total_nodes=context_total_path,
                           alpha=1.0,
                           chunksize=batch_size)
        io_utils.save_embedding(model.node_embedding, file_name=output_file+"_lr-{}".format(lr))