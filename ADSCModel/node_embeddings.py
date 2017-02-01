__author__ = 'ando'

import logging
import time
import threading
from queue import Queue
import numpy as np
from utils.embedding import chunkize_serial, RepeatCorpusNTimes, train_sg

logger = logging.getLogger()



class Node2Vec(object):
    def __init__(self, workers=1, alpha=0.1, min_alpha=0.0001, negative=0):
        self.workers = workers
        self.alpha = float(alpha)
        self.min_alpha = min_alpha
        self.negative = negative
        self.window_size = 1

    def train(self, model, edges, _lambda2=0.0, total_node=None, word_count=0, chunksize=100, iter = 1):
        """
        Update the model's neural weights from a sequence of paths (can be a once-only generator stream).
        """
        logger.info("training model with %i workers on %i vocabulary and %i features and 'negative sampling'=%s" %
                    (self.workers, len(model.vocab), model.layer1_size, self.negative))

        if not model.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        edges = RepeatCorpusNTimes(edges, iter)
        total_node = edges.total_node

        start, next_report = time.time(), [1.0]
        word_count = [word_count]
        loss = []

        #int(sum(v.count * v.sample_probability for v in self.vocab.values()))
        jobs = Queue(maxsize=2*self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)


        def worker_train():
            """Train the model, lifting lists of paths from the jobs queue."""
            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break

                py_work = np.zeros(model.layer1_size)
                py_work_o3 = np.zeros(model.layer1_size)
                py_work1_o3 = np.zeros(model.layer1_size)
                py_work2_o3 = np.zeros(model.layer1_size ** 2)
                # update the learning rate before every job
                alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count[0] / total_node))
                # how many words did we train on? out-of-vocabulary (unknown) words do not count
                job_words = 0
                job_loss = 0

                for path in job:
                    words_path, loss_path = train_sg(model.node_embedding, model.node_embedding, path, alpha, self.negative, self.window_size, model.table,
                                             py_centroid=model.centroid, py_inv_covariance_mat=model.inv_covariance_mat, py_pi=model.pi, py_k=model.k, py_covariance_mat=model.covariance_mat,
                                             py_lambda1=1.0, py_lambda2=_lambda2,
                                             py_size=model.layer1_size, py_work=py_work, py_work_o3=py_work_o3, py_work1_o3=py_work1_o3, py_work2_o3=py_work2_o3,
                                             py_is_node_embedding=1)
                    job_words += words_path
                    job_loss += loss_path/(len(path) * ((self.window_size*2)-1))

                # job_words = sum(train_sg(model.node_embedding, model.node_embedding, path, alpha, self.negative, self.window_size, model.table,
                #                          py_centroid=model.centroid, py_inv_covariance_mat=model.inv_covariance_mat, py_pi=model.pi, py_k=model.k, py_covariance_mat=model.covariance_mat,
                #                          py_lambda1=1.0, py_lambda2=_lambda2,
                #                          py_size=model.layer1_size, py_work=py_work, py_work_o3=py_work_o3, py_work1_o3=py_work1_o3, py_work2_o3=py_work2_o3,
                #                          py_is_node_embedding=1) for path in job)


                with lock:
                    word_count[0] += job_words
                    loss.append(job_loss)

                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        logger.info("PROGRESS: at %.2f%% words, alpha %.05f, %.0f words/s" %
                                    (100.0 * word_count[0] / total_node, alpha, word_count[0] / elapsed if elapsed else 0.0))
                        logger.info('loss: %f' % np.mean(loss))
                        next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports



        workers = [threading.Thread(target=worker_train) for _ in range(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        def prepare_sentences():
            for edge in edges:
                # avoid calling random_sample() where prob >= 1, to speed things up a little:
                sampled = [model.vocab[word] for word in edge
                           if word in model.vocab and (model.vocab[word].sample_probability >= 1.0 or model.vocab[word].sample_probability >= np.random.random_sample())]
                yield sampled

        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(chunkize_serial(prepare_sentences(), chunksize)):
            # logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
            jobs.put(job)
        logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
        for _ in range(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        logger.warning("training on %i words took %.1fs, %.0f words/s" %
                    (word_count[0], elapsed, word_count[0] / elapsed if elapsed else 0.0))
        loss = np.mean(loss)
        logging.info('LOSS: %f' % loss)
        return loss