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
    input_dataset = "Wikipedia"
    walks_filebase = os.path.join('data', input_dataset)            # where read/write the sampled path
    walk_files = ["{}_n2v.walks.0{}".format(os.path.join(walks_filebase, input_dataset), i) for i in range(10) if os.path.isfile("{}_n2v.walks.0{}".format(os.path.join(walks_filebase, input_dataset), i))]

    for file_idx, _file in enumerate(walk_files):
        with open(os.path.join(walks_filebase, "n2v_path_p-4_q-1", "{}_n2v.walks.0{}".format(input_dataset, file_idx)), 'w') as writer:
            with open(_file, 'r') as file:
                for line in file:
                    writer.write("{}\n".format(" ".join(map(lambda x: str(int(x) - 1), line.strip().split(" ")))))