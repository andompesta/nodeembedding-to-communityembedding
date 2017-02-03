__author__ = 'ando'
import logging
import numpy as np
import pickle
from os.path import exists
from os import makedirs
from utils.embedding import Vocab
from utils.IO_utils import load_ground_true


logger = logging.getLogger('adsc')

class Model(object):
    '''
    class that keep track of all the parameters used during the learning of the embedding.
    '''


    def __init__(self, G=None, size=2, min_count=0, downsampling=0, seed=1, table_size=100000000,
                 vocab=None, index2word=None,
                 node_embedding=None,
                 context_embedding=None,
                 community_embedding=None,
                 inv_covariance_mat=None,
                 pi=None,
                 path_labels='data/',
                 input_file=None,
                 vocabulary_counts=None):
        '''
        :param G: graph used for the computation
        :param size: projection space
        :param min_count: ignore all words with total frequency lower than this.
        :param downsampling: perform downsampling of common node
        :param seed: seed for random function
        :param table_size: size of the negative sampling table
        :param vocab: dictionary between a node and its count in the paths
        :param index2word: index between a node and its representation

        :param node_embedding: matrix containing the node embedding
        :param context_embedding: matrix containing the context embedding
        :param community_embedding: matrix containing the community embedding
        :param inv_covariance_mat: matrix representing the covariance matrix of the mixture clustering
        :param pi: probability distribution of each node respect the communities

        :param path_labels: location of the file containing the ground true (label for each node)
        :param input_file: name of the file containing the ground true (label for each node)
        :param vocabulary_counts: number of time a node is sampled
        :return:
        '''
        if vocab is None:
            self.vocab = {}  # mapping from a word (string) to a Vocab object
        else:
            self.vocab = vocab

        if index2word is None:
            self.index2word = []  # map from a word's matrix index (int) to word (string)
        else:
            self.index2word = index2word


        self.min_count = min_count
        self.downsampling = downsampling
        self.seed = seed
        np.random.seed(self.seed)
        self.loss = 0

        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.layer1_size = int(size)
        self.table_size = table_size

        if context_embedding is not None:
            self.context_embedding = context_embedding
        if node_embedding is not None:
            self.node_embedding = node_embedding
        if inv_covariance_mat is not None:
            self.inv_covariance_mat = inv_covariance_mat
        if community_embedding is not None:
            self.community_embedding = community_embedding
        if pi is not None:
            self.pi = pi

        if G is not None:
            self.build_vocab_(vocabulary_counts)
            self.ground_true, self.k = load_ground_true(path=path_labels, file_name=input_file)
            # inizialize node and context embeddings
            self.reset_weights()
            self.make_table()

    def build_vocab_(self, vocab):
        """
        Build vocabulary from a sequence of paths (can be a once-only generator stream).
        """
        # assign a unique index to each word
        self.vocab, self.index2word = {}, []
        for word, count in vocab.items():
            v = Vocab()
            v.count = count
            if v.count >= self.min_count:
                v.index = len(self.vocab)
                self.index2word.append(word)
                self.vocab[word] = v
        logger.debug("total %i node after removing those with count<%s" % (len(self.vocab), self.min_count))

        # precalculate downsampling thresholds
        self.precalc_sampling()


    def precalc_sampling(self):
        '''
            Peach vocabulary item's threshold for sampling
        '''

        if self.downsampling:
            logger.info("frequent-word downsampling, threshold %g; progress tallies will be approximate" % (self.downsampling))
            total_words = sum(v.count for v in self.vocab.values())
            threshold_count = float(self.downsampling) * total_words

        for v in self.vocab.values():
            prob = (np.sqrt(v.count / threshold_count) + 1) * (threshold_count / v.count) if self.downsampling else 1.0
            v.sample_probability = min(prob, 1.0)

    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.debug("resetting layer weights")

        self.node_embedding = np.empty((len(self.vocab), self.layer1_size), dtype=np.float32)

        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in range(len(self.vocab)):
            self.node_embedding[i] = (np.random.rand(self.layer1_size) - 0.5) / self.layer1_size

        self.context_embedding = np.zeros((len(self.vocab), self.layer1_size), dtype=np.float32)
        self.centroid = np.zeros((self.k, self.layer1_size), dtype=np.float32)
        self.covariance_mat = np.zeros((self.k, self.layer1_size), dtype=np.float32)
        self.inv_covariance_mat = np.zeros((self.k, self.layer1_size), dtype=np.float32)
        self.pi = np.zeros(self.k, dtype=np.float32)




    def make_table(self, power=0.75):
        """
        Create a table using stored vocabulary word counts for drawing random words in the negative
        sampling training routines.

        Called internally from `build_vocab()`.

        """
        logger.info("constructing a table with noise distribution from %i words" % len(self.vocab))
        # table (= list of words) of noise distribution for negative sampling
        vocab_size = len(self.index2word)
        self.table = np.zeros(self.table_size, dtype=np.uint32)

        if not vocab_size:
            logger.warning("empty vocabulary in word2vec, is this intended?")
            return

        # compute sum of all power (Z in paper)
        train_words_pow = float(sum([self.vocab[word].count**power for word in self.vocab]))
        # go through the whole table and fill it up with the word indexes proportional to a word's count**power
        widx = 0
        # normalize count^0.75 by Z
        d1 = self.vocab[self.index2word[widx]].count**power / train_words_pow
        for tidx in range(self.table_size):
            self.table[tidx] = widx
            if 1.0 * tidx / self.table_size > d1:
                widx += 1
                d1 += self.vocab[self.index2word[widx]].count**power / train_words_pow
            if widx >= vocab_size:
                widx = vocab_size - 1
        logger.debug('max table %d' % max(self.table))



    def save(self, path='data', file_name=None):
        if not exists(path):
            makedirs(path)

        with open(path + '/' + file_name + '.bin', 'wb') as file:
            pickle.dump(self.__dict__, file)

    def load_model(self, path='data', file_name=None):
        with open(path + '/' + file_name + '.bin', 'rb') as file:
            self.__dict__ = pickle.load(file)
            logger.info('model loaded , size: %d \t table_size: %d \t down_sampling: %.5f \t communities %d' % (self.layer1_size, self.table_size, self.downsampling, self.k))
            return self
