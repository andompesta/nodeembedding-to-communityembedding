__author__ = 'ando'

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from os.path import exists
from os import makedirs
import numpy as np
import itertools


# plt.rc('xtick', labelsize=20)
# plt.rc('ytick', labelsize=20)

STARTING_NODE = 16
def graph_plot(G, path="graph", graph_name='graph', save=True, show=True):
    spring_pos = nx.spring_layout(G)
    color_map = list(plt.cm.rainbow(np.linspace(0.1, 0.9, G.number_of_nodes())))

    order_edges = nx.bfs_edges(G, G.nodes()[STARTING_NODE])
    node_color = np.zeros((G.number_of_nodes(), 4))

    for index, d in enumerate(order_edges):
        node_color[d[1]-1] = color_map[index+1]

    node_color[STARTING_NODE] = color_map[0]


    plt.figure(figsize=(5, 5))
    plt.axis("off")


    nx.draw_networkx(G, node_color=node_color, pos=spring_pos)

    if save:
        if not exists(path):
            makedirs(path)
        plt.savefig(path + '/' + graph_name +'.png')
        plt.close()

    elif show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    return node_color


def node_space_plot_3D(embedding, path="graph", graph_name='graph', save=True, color_values=[]):
    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111, projection='3d')


    ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], c=color_values, cmap=cm.Spectral, marker='o', s=70)
    for node in range(len(embedding)):

        ax.text(embedding[node][0], embedding[node][1], embedding[node][2],  '%s' % (str(node)), size=10)


    if save:
        if not exists(path):
            makedirs(path)
        plt.savefig(path + '/' + graph_name + '_prj_3d' + '.png')
        plt.close()
    else:
        plt.show()

def node_space_plot_2D(embedding, path="graph", graph_name='graph', save=True, color_values=[], centroid=None):


    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    color_dict = {1:'r', 2:'y', 3:'b', 4:'g'}
    nodes_id = np.array(list(range(1, len(embedding)+1)))
    data = np.concatenate((embedding, nodes_id.reshape(len(nodes_id), 1)), axis=1)

    # data = sorted(data, key=lambda row: row[2])

    # color_nodes = {}
    # for node_id, node_color in enumerate(color_values):
    #     if node_color in color_nodes:
    #         color_nodes[node_color].append(node_id)
    #     else:
    #         color_nodes[node_color] = [node_id]
    #
    # order_embedding = np.zeros((embedding.shape[0], embedding.shape[1] + 1))
    #
    # i = 0
    # for color in [1, 2, 3, 4]:
    #     for node in color_nodes[color]:
    #         order_embedding[i, :2] = embedding[node]
    #         order_embedding[i, -1] = color
    #         i += 1
    # ax.scatter(order_embedding[:,0], order_embedding[:,1], c=order_embedding[:, 2], cmap=cm.Spectral, marker='o', s=100)

    for node in data:
        ax.scatter(node[0], node[1], c=color_dict[color_values[node[2]-1]], marker='o', s=100)
        ax.text(node[0], node[1],  '%s' % (str(int(node[2]))), size=10)

    if centroid is not None:
        ax.scatter(centroid[:,0], centroid[:,1], marker='x', c='r')

    if save:
        if not exists(path):
            makedirs(path)
        plt.savefig(path + graph_name + '_prj_2d' + '.png')
        plt.close()
    else:
        plt.show()