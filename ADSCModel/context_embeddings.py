__author__ = 'ando'

import logging
import time
import threading
from queue import Queue
import numpy as np
from utils.embedding import train_sg, chunkize_serial, prepare_sentences
from scipy.special import expit as sigmoid

class Context2Vec(object):
    '''
    Class that train the context embedding
    '''
    def __init__(self, alpha=0.1, window_size=5, workers=1, min_alpha=0.0001, negative=5):
        '''
        :param alpha: learning rate
        :param window: windows size used to compute the context embeddings
        :param workers: number of thread
        :param min_alpha: min learning rate
        :param negative: number of negative samples
        :return:
        '''

        self.alpha = float(alpha)
        self.workers = workers
        self.min_alpha = min_alpha
        self.negative = negative
        self.window_size = int(window_size)


    def train(self, model, paths, total_words, _lambda1=1.0, _lambda2=0.0, word_count=0, chunksize=150):
        """
        Update the model's neural weights from a sequence of paths (can be a once-only generator stream).
        """
        print("training model with %i workers on %i vocabulary and %i features and 'negative sampling'=%s" %
                    (self.workers, len(model.vocab), model.layer1_size, self.negative))

        if not model.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        start, next_report = time.time(), [1.0]
        if total_words is None:
            raise AttributeError('need to the the number of node')

        word_count = [word_count]
        jobs = Queue(maxsize=2*self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)

        def worker_train():
            """Train the model, lifting lists of paths from the jobs queue."""
            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break

                job_words = 0

                py_work = np.zeros(model.layer1_size, dtype=np.float32)
                py_work_o3 = np.zeros(model.layer1_size, dtype=np.float32)
                py_work1_o3 = np.zeros(model.layer1_size, dtype=np.float32)
                py_work2_o3 = np.zeros(model.layer1_size ** 2, dtype=np.float32)
                # update the learning rate before every job
                # alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count[0] / total_words))
                # how many words did we train on? out-of-vocabulary (unknown) words do not count

                if _lambda1 > 0:
                    job_words = sum(train_sg(model.node_embedding, model.context_embedding, path, self.alpha, self.negative, self.window_size, model.table,
                                                  py_centroid=model.centroid, py_inv_covariance_mat=model.inv_covariance_mat, py_pi=model.pi, py_k=model.k, py_covariance_mat=model.covariance_mat,
                                                  py_lambda1=_lambda1, py_lambda2=_lambda2, py_size=model.layer1_size,
                                                  py_work=py_work, py_work_o3=py_work_o3, py_work1_o3=py_work1_o3, py_work2_o3=py_work2_o3, py_is_node_embedding=0) for path in job) #execute the sgd

                with lock:
                    word_count[0] += job_words
                    # loss[0] += job_loss

                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        print("PROGRESS: at %.2f%% words, alpha %.05f, %.0f words/s" %
                                    (100.0 * word_count[0] / total_words, self.alpha, word_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

        workers = [threading.Thread(target=worker_train) for _ in range(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(chunkize_serial(prepare_sentences(model, paths), chunksize)):
            jobs.put(job)

        print("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
        for _ in range(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        print("training on %i words took %.1fs, %.0f words/s" %
                    (word_count[0], elapsed, word_count[0] / elapsed if elapsed else 0.0))
