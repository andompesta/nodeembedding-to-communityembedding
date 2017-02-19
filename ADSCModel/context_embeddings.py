__author__ = 'ando'

import logging
import time
import threading
from queue import Queue
import numpy as np
from utils.embedding import train_sg, chunkize_serial, prepare_sentences
from scipy.special import expit as sigmoid

try:
    raise(ImportError)
    from utils.loss_inner import o2_loss, FAST_VERSION
    print('Fast version ' + str(FAST_VERSION))
except ImportError as e:
    print(e)
    def o2_loss(node_embedding, py_negative_embedding, py_path, py_negative, py_window, py_table, py_lambda=1.0, py_size=None):
        ret_loss = 0
        for pos, node in enumerate(py_path):  # node = input vertex of the system
            if node is None:
                continue  # OOV node in the input path => skip

            # labels = np.zeros(py_negative + 1)
            labels = 1.0  # frist node come from the path, the other not (lable[1:]=0)

            start = max(0, pos - py_window)
            # now go over all words from the (reduced) window, predicting each one in turn
            for pos2, node2 in enumerate(py_path[start: pos + py_window + 1],
                                         start):  # node 2 are the output nodes predicted form node
                # don't train on OOV words and on the `word` itself
                if node2 and not (pos2 == pos):
                    positive_node_embedding = node_embedding[node2.index]  # correct node embeddings
                    negative_nodes_embedding = py_negative_embedding[node.index]
                    fb = sigmoid(np.dot(negative_nodes_embedding, positive_node_embedding))  # propagate hidden -> output
                    gb = (labels - fb)
                    ret_loss -= np.log(gb)
        return ret_loss * py_lambda

logger = logging.getLogger()

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
        # lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)
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
        # # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
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
                    # for path in job:
                    #     words_done, loss_path = train_sg(model.node_embedding, model.context_embedding, path, alpha, self.negative, self.window_size, model.table,
                    #              py_centroid=model.centroid, py_inv_covariance_mat=model.inv_covariance_mat, py_pi=model.pi, py_k=model.k, py_covariance_mat=model.covariance_mat,
                    #              py_lambda1=_lambda1, py_lambda2=_lambda2, py_size=model.layer1_size,
                    #              py_work=py_work, py_work_o3=py_work_o3, py_work1_o3=py_work1_o3, py_work2_o3=py_work2_o3, py_is_node_embedding=0)
                    #
                    #     job_words += words_done
                    #     job_loss += loss_path

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
