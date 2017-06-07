__author__ = 'ando'
import logging as log

import time
import threading
from queue import Queue
import numpy as np
from utils.embedding import train_sg, chunkize_serial, prepare_sentences
from scipy.special import expit as sigmoid


log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)

try:
    from utils.training_sdg_inner import train_o2, FAST_VERSION
    log.info('cython version {}'.format(FAST_VERSION))
except ImportError as e:
    log.error(e)

def o2_loss(node_embedding, py_negative_embedding, py_path, py_negative, py_window, py_table, py_lambda=1.0, py_size=None):
    ret_loss = 0
    for pos, node in enumerate(py_path):  # node = input vertex of the system
        if node is None:
            continue  # OOV node in the input path => skip

        # labels = np.zeros(py_negative + 1)
        labels = 1.0  # frist node come from the path, the other not (lable[1:]=0)

        start = max(0, pos - py_window)
        # now go over all nodes from the (reduced) window, predicting each one in turn
        for pos2, node2 in enumerate(py_path[start: pos + py_window + 1],
                                     start):  # node 2 are the output nodes predicted form node
            # don't train on OOV nodes and on the `node` itself
            if node2 and not (pos2 == pos):
                positive_node_embedding = node_embedding[node2.index]  # correct node embeddings
                negative_nodes_embedding = py_negative_embedding[node.index]
                fb = sigmoid(np.dot(negative_nodes_embedding, positive_node_embedding))  # propagate hidden -> output
                gb = (labels - fb)
                ret_loss -= np.log(gb)
    return ret_loss * py_lambda


class Context2Vec(object):
    '''
    Class that train the context embedding
    '''
    def __init__(self, lr=0.1, window_size=5, workers=1, negative=5):
        '''
        :param lr: learning rate
        :param window: windows size used to compute the context embeddings
        :param workers: number of thread
        :param negative: number of negative samples
        :return:
        '''

        self.lr = float(lr)
        self.workers = workers
        self.negative = negative
        self.window_size = int(window_size)

    def loss(self, model, paths, total_paths, _lambda1=1.0):
        start, next_report, num_paths, loss = time.time(), 5.0, 0.0, 0.0

        def worker_loss(job, num_paths, next_report):
            """Train the model, lifting lists of paths from the jobs queue."""
            job_loss = sum([o2_loss(model.node_embedding, model.context_embedding, path, self.negative, self.window_size, model.table, _lambda1, model.layer1_size) for path in job]) #execute the sgd
            num_paths += len(job)
            elapsed = time.time() - start

            if elapsed >= next_report:
                print("PROGRESS: at %.2f%% path, %.0f paths/s" %(100.0 * num_paths/total_paths, num_paths / elapsed if elapsed else 0.0))
                next_report = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

            return job_loss, num_paths, next_report

        for job_no, job in enumerate(chunkize_serial(prepare_sentences(model, paths), 250)):
            job_loss, num_paths, next_report = worker_loss(job, num_paths, next_report)
            loss += job_loss
        return loss

        # jobs = Queue(
        #     maxsize=2 * self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        # lock = threading.Lock()  # for shared state (=number of nodes trained so far, log reports...)
        # start, next_report, num_paths, loss = time.time(), [5.0], [0.0], [0.0]


        # def worker_loss():
        #     """Train the model, lifting lists of paths from the jobs queue."""
        #     while True:
        #         job = jobs.get(block=True)
        #         if job is None:  # data finished, exit
        #             jobs.task_done()
        #             logger.debug('thread %s break' % threading.current_thread().name)
        #             break
        #
        #         job_loss = sum(o2_loss(model.node_embedding, model.context_embedding, path, self.negative, self.window_size, model.table, _lambda1, model.layer1_size) for path in job) #execute the sgd
        #
        #         jobs.task_done()
        #         lock.acquire(timeout=30)
        #         try:
        #             loss[0] += job_loss
        #             num_paths[0] += len(job)
        #
        #             elapsed = time.time() - start
        #             if elapsed >= next_report[0]:
        #                 print("PROGRESS: at %.2f%% path, %.0f paths/s" %
        #                             (100.0 * num_paths[0]/total_paths, num_paths[0] / elapsed if elapsed else 0.0))
        #                 next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports
        #         finally:
        #             lock.release()
        #
        # workers = [threading.Thread(target=worker_loss, name='thread_loss_' + str(i)) for i in range(self.workers)]
        # for thread in workers:
        #     thread.daemon = True  # make interrupting the process with ctrl+c easier
        #     thread.start()
        #
        # # convert input strings to Vocab objects (eliding OOV/downsampled nodes), and start filling the jobs queue
        # for job_no, job in enumerate(chunkize_serial(prepare_sentences(model, paths), 250)):
        #     # logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
        #     jobs.put(job)
        #
        # for _ in range(self.workers):
        #     jobs.put(None)  # give the workers heads up that they can finish -- no more work!
        #
        # for thread in workers:
        #     thread.join()
        # return loss[0]


    def train(self, model, paths, total_nodes, alpha=1.0, node_count=0, chunksize=150):
        """
        Update the model's neural weights from a sequence of paths (can be a once-only generator stream).

        :param model: model containing the shared data
        :param paths: generator of the paths
        :param total_nodes: total number of nodes in the path
        :param alpha: trade-off parameter
        :param node_count: init of the number of nodes
        :param chunksize: size of the batch
        :return:
        """
        assert model.node_embedding.dtype == np.float32
        assert model.context_embedding.dtype == np.float32
        log.info("O2 training model with %i workers on %i vocabulary and %i features and 'negative sampling'=%s" %
                    (self.workers, len(model.vocab), model.layer1_size, self.negative))

        if alpha <= 0.:
            return

        if not model.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        start, next_report = time.time(), [1.0]
        if total_nodes is None:
            raise AttributeError('need the number of node')

        node_count = [node_count]
        jobs = Queue(maxsize=2*self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()  # for shared state (=number of nodes trained so far, log reports...)

        def worker_train():
            """Train the model, lifting lists of paths from the jobs queue."""
            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break

                job_nodes = 0

                py_work = np.zeros(model.layer1_size, dtype=np.float32)
                # update the learning rate before every job
                # alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * node_count[0] / total_nodes))
                # how many nodes did we train on? out-of-vocabulary (unknown) nodes do not count

                if alpha > 0.:
                    job_nodes = sum(train_o2(model.node_embedding, model.context_embedding, path, self.lr, self.negative, self.window_size, model.table,
                                             py_alpha=alpha, py_size=model.layer1_size, py_work=py_work) for path in job) #execute the sgd

                with lock:
                    node_count[0] += job_nodes

                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        print("PROGRESS: at %.2f%% nodes, alpha %.05f, %.0f nodes/s" %
                                    (100.0 * node_count[0] / total_nodes, self.alpha, node_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

        workers = [threading.Thread(target=worker_train) for _ in range(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        # convert input strings to Vocab objects (eliding OOV/downsampled nodes), and start filling the jobs queue
        for job_no, job in enumerate(chunkize_serial(prepare_sentences(model, paths), chunksize)):
            jobs.put(job)

        log.debug("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
        for _ in range(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        log.info("training on %i nodes took %.1fs, %.0f nodes/s" %
                    (node_count[0], elapsed, node_count[0] / elapsed if elapsed else 0.0))
