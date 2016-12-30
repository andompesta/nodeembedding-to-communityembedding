__author__ = 'ando'

import pickle
import numpy as np

def save_ground_true(path, community_color):
    with open(path, 'w') as txt_file:
        for node, com in enumerate(community_color):
            txt_file.write('%d\t%d\n' % ((node+1), com))

def load_ground_true(path='data/', file_name=None):
    temp = []
    with open(path + file_name + '.labels', 'r') as file:
        for line_no, line in enumerate(file):
            tokens = line.strip().split('\t')
            temp.append(int(tokens[1]))
    ground_true = np.array(temp, dtype=np.uint8)
    k = max(ground_true)
    return ground_true, k



def save_embedding(embeddings, file_name, path='data'):
    with open(path + '/' + file_name + '.txt', 'w') as file:
        for node_id, embed in enumerate(embeddings):
            file.write(str(node_id+1) + '\t' + " ".join([str(val) for val in embed]) + '\n')

def load_embedding(file_name, path='data'):
    ret = []
    with open(path + '/' + file_name + '.txt', 'r') as file:
        for line in file:
            tokens = line.strip().split('\t')
            node_values = [float(val) for val in tokens[1].strip().split(' ')]
            ret.append(node_values)
    return np.array(ret)


def load_embedding_line(file_name, path='/Users/ando/Project/line/graph'):
    ret = []
    file_data = {}
    with open(path + '/' + file_name + '.txt', 'r') as file:
        for line in file:

            tokens = line.strip().split('\t')
            node_id = int(tokens[0])
            values = [float(token) for token in tokens[1].split()]
            file_data[node_id] = values

    ret = [file_data[key] for key in sorted(file_data.keys())]
    return np.array(ret)

def save_membership(membership_matrix, file_name, path='data'):
    with open(path + '/' + file_name + '.txt', 'w') as file:
        for node_id, value in enumerate(membership_matrix):
            file.write('%d\t%d\n' % (node_id, value))

def load_membership(file_name, path='data'):
    membership = []
    with open(path + '/' + file_name + '.txt', 'r') as file:
        for line in file:
            tokens = line.strip().split('\t')
            membership.append(int(tokens[1]))
    return membership


def save(node_embedding, file_name, path='data'):
    with open(path + '/' + file_name + '.embedding', 'wb') as file:
        pickle.dump(node_embedding, file)

def load_model(file_name, path='data'):
    with open(path + '/' + file_name + '.txt', 'rb') as file:
        model = pickle.load(file)
        return model
