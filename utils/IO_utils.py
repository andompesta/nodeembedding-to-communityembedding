__author__ = 'ando'

import pickle
import numpy as np
from os.path import join as path_join, dirname
from os import makedirs

def save_ground_true(file_name, community_color, path="./data",):
    '''
    :param file_name: name of the file
    :param path: path where to save the data
    :param community_color: list of len(nodes) with the color values of each node
    '''
    with open(path_join(path, "{}.txt".format(file_name)), 'w') as txt_file:
        for node, com in enumerate(community_color):
            txt_file.write('%d\t%d\n' % ((node+1), com))

def load_ground_true(path='data/', file_name=None, multilabel=False):
    '''
    Return the label and the number of communities of the dataset
    :param path: path to the dir containing the file
    :param file_name: filename to read
    :param multilabel: True if the dataset is multilabel
    :return:
    '''
    labels = {}
    max = 0
    with open(path_join(path, file_name + '.labels'), 'r') as file:
        for line_no, line in enumerate(file):
            tokens = line.strip().split('\t')
            node_id = int(tokens[0])
            label_id = int(tokens[1])
            if label_id > max:
                max = label_id
            if node_id in labels:
                labels[node_id].append(label_id)
            else:
                labels[node_id] = [label_id]


    ret = []
    for key in sorted(labels.keys()):
        if multilabel:
            ret.append(labels[key])
        else:
            ret.append(labels[key][0])
    return ret, max

def save_embedding(embeddings, file_name, path='data'):
    '''
    save the final embedding as a txt file
    file structure = <node_id>\t<feautre_1>\s<feautre_2>...
    :param embeddings: embedding to save
    :param file_name: file_name
    :param path: directory where to save the data
    '''
    full_path = path_join(path, file_name + '.txt')
    makedirs(dirname(full_path), exist_ok=True)

    with open(full_path, 'w') as file:
        for node_id, embed in enumerate(embeddings):
            file.write(str(node_id+1) + '\t' + " ".join([str(val) for val in embed]) + '\n')

def load_embedding(file_name, path='data'):
    '''
    Load an node embedding form a txt file
     file structure = <node_id>\t<feautre_1>\s<feautre_2>...
    :param file_name: file name to load
    :param path: directory where the file is saved
    :return: node_embedding as numpy matrix
    '''
    full_path = path_join(path, file_name + '.txt')
    ret = []
    with open(full_path, 'r') as file:
        for line in file:
            tokens = line.strip().split('\t')
            node_values = [float(val) for val in tokens[1].strip().split(' ')]
            ret.append(node_values)
    return np.array(ret)


def save(data, file_name, path='data'):
    '''
    Dump datastructure with pickle
    :param data: data to dump
    :param file_name: file name
    :param path: dire where to save the file
    :return:
    '''
    full_path = path_join(path, file_name + '.bin')
    makedirs(dirname(full_path), exist_ok=True)
    with open(full_path, 'wb') as file:
        pickle.dump(data, file)


def load(file_name, path='data'):
    '''
    Load datastructure with pickle
    :param file_name: file name
    :param path: dire where to save the file
    :return:
    '''
    full_path = path_join(path, file_name + '.bin')
    with open(full_path, 'rb') as file:
        data = pickle.load(file)
    return data