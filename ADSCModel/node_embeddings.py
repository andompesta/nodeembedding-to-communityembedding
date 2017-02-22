__author__ = 'ando'

import logging
import time
import threading
from queue import Queue
import numpy as np
from utils.embedding import chunkize_serial, RepeatCorpusNTimes, train_sg, prepare_sentences
from itertools import islice, chain, zip_longest
from scipy.special import expit as sigmoid

class Node2Vec(object):
    def __init__(self, workers=1, alpha=0.1, min_alpha=0.0001, negative=0, ):

        self.workers = workers
        self.alpha = float(alpha)
        self.min_alpha = min_alpha
        self.negative = negative
        self.window_size = 1



    def train(self, model, edges, _lambda1=1.0, _lambda2=0.0, total_node=None, chunksize=150, iter = 1):
        """
        Update the model's neural weights from a sequence of paths (can be a once-only generator stream).
        """
        print("training model with %i workers on %i vocabulary and %i features and 'negative sampling'=%s" %
                    (self.workers, len(model.vocab), model.layer1_size, self.negative))

        if not model.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        edges = RepeatCorpusNTimes(edges, iter)
        total_node = edges.total_node
        print('total edges: %d' % total_node)
        start, next_report, word_count = time.time(), [5.0], [0]


        #int(sum(v.count * v.sample_probability for v in self.vocab.values()))
        jobs = Queue(maxsize=2*self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()

        # def learn_single(job, word_count):
        #     py_work = np.zeros(model.layer1_size)
        #     py_work_o3 = np.zeros(model.layer1_size)
        #     py_work1_o3 = np.zeros(model.layer1_size)
        #     py_work2_o3 = np.zeros(model.layer1_size ** 2)
        #     # update the learning rate before every job
        #     alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count/ total_node))
        #     # how many words did we train on? out-of-vocabulary (unknown) words do not count
        #
        #
        #     job_words = sum(train_sg(model.node_embedding, model.node_embedding, path, alpha, self.negative, self.window_size, model.table,
        #                              py_centroid=model.centroid, py_inv_covariance_mat=model.inv_covariance_mat, py_pi=model.pi, py_k=model.k, py_covariance_mat=model.covariance_mat,
        #                              py_lambda1=_lambda1, py_lambda2=_lambda2,
        #                              py_size=model.layer1_size, py_work=py_work, py_work_o3=py_work_o3, py_work1_o3=py_work1_o3, py_work2_o3=py_work2_o3,
        #                              py_is_node_embedding=1) for path in job if path is not None)
        #
        #
        #
        #     elapsed = time.time() - start
        #     if elapsed >= next_report[0]:
        #         word_count += job_words
        #         print("PROGRESS: at %.2f%% words, alpha %.05f, %.0f words/s" %
        #                     (100.0 * word_count / total_node, alpha, word_count / elapsed if elapsed else 0.0))
        #         next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports
        #     return word_count
        # for job_no, job in enumerate(batch_generator(prepare_sentences(edges), chunksize)):
        #     # logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
        #     word_count = learn_single(job, word_count)


        def worker_train():
            """Train the model, lifting lists of paths from the jobs queue."""
            while True:
                job = jobs.get(block=True)
                if job is None:  # data finished, exit
                    jobs.task_done()
                    # print('thread %s break' % threading.current_thread().name)
                    break


                py_work = np.zeros(model.layer1_size)
                py_work_o3 = np.zeros(model.layer1_size)
                py_work1_o3 = np.zeros(model.layer1_size)
                py_work2_o3 = np.zeros(model.layer1_size ** 2)
                # update the learning rate before every job
                # alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count[0] / total_node))
                # how many words did we train on? out-of-vocabulary (unknown) words do not count


                job_words = sum(train_sg(model.node_embedding, model.node_embedding, path, self.alpha, self.negative, self.window_size, model.table,
                                         py_centroid=model.centroid, py_inv_covariance_mat=model.inv_covariance_mat, py_pi=model.pi, py_k=model.k, py_covariance_mat=model.covariance_mat,
                                         py_lambda1=_lambda1, py_lambda2=_lambda2,
                                         py_size=model.layer1_size, py_work=py_work, py_work_o3=py_work_o3, py_work1_o3=py_work1_o3, py_work2_o3=py_work2_o3,
                                         py_is_node_embedding=1) for path in job if path is not None)
                jobs.task_done()
                lock.acquire(timeout=30)
                try:
                    word_count[0] += job_words

                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        print("PROGRESS: at %.2f%% words\tword_computed %d\talpha %.05f\t %.0f words/s" %
                                    (100.0 * word_count[0] / total_node, word_count[0], self.alpha, word_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 5.0  # don't flood the log, wait at least a second between progress reports
                finally:
                    lock.release()



        workers = [threading.Thread(target=worker_train, name='thread_'+str(i)) for i in range(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()


        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(chunkize_serial(prepare_sentences(model, edges), chunksize)):
            jobs.put(job)


        for _ in range(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        print("training on %i words took %.1fs, %.0f words/s" %
                    (word_count[0], elapsed, word_count[0]/ elapsed if elapsed else 0.0))
