__author__ = 'ando'

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from os.path import exists
from os import makedirs
import numpy as np
import itertools
import math

# plt.rc('xtick', labelsize=20)
# plt.rc('ytick', labelsize=20)

STARTING_NODE = 16
def graph_plot(G, path="graph", graph_name='graph', nodes_color=None,save=True, show=True):
    spring_pos = nx.spring_layout(G)

    plt.figure(figsize=(5, 5))
    plt.axis("off")

    if nodes_color == None:
        color_map = list(plt.cm.rainbow(np.linspace(0.1, 0.9, G.number_of_nodes())))
        order_edges = nx.bfs_edges(G, G.nodes()[STARTING_NODE])
        nodes_color = np.zeros((G.number_of_nodes(), 4))

        for index, d in enumerate(order_edges):
            nodes_color[d[1]-1] = color_map[index+1]

        nodes_color[STARTING_NODE] = color_map[0]
        nx.draw_networkx(G, node_color=nodes_color, pos=spring_pos)

    else:
        color_map = {0:'lightcoral', 1:'yellow', 2:'limegreen', 3:'cyan'}
        for community in range(4):
            nodes_in_community = np.where(nodes_color == community + 1)[0] + 1
            nx.draw_networkx_nodes(G, spring_pos,
                                   nodelist=nodes_in_community.tolist(),
                                   node_color=color_map[community],
                                   node_size=400,
                                   alpha=0.9)

        nx.draw_networkx_edges(G, spring_pos, width=1.0, alpha=0.5)
        labels = {}
        for node in G.nodes():
            labels[node] = node
        nx.draw_networkx_labels(G,spring_pos, labels,font_size=14)

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

    return nodes_color


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

def node_space_plot_2D(embedding, path="graph", graph_name='graph', save=True, color_values=[], centroid=None, title=None, grid=False):
    color_map = {0:'lightcoral', 1:'yellow', 2:'limegreen', 3:'cyan'}
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    nodes_id = np.array(list(range(1, len(embedding)+1)))
    data = np.concatenate((embedding, nodes_id.reshape(len(nodes_id), 1), color_values.reshape(len(color_values), 1)), axis=1)

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
        ax.scatter(node[0], node[1], c=color_map[node[3]-1], marker='o', s=100, alpha=0.8)
        ax.text(node[0], node[1],  '%s' % (str(int(node[2]))), size=10)

    if centroid is not None:
        ax.scatter(centroid[:,0], centroid[:,1], marker='x', c='r')

    if grid:
        x_max, x_min = -0.5, -4.0
        y_max, y_min = 3.2, -3

        x_step = (x_max - x_min) / 4.0
        y_step = (y_max - y_min) / 4.0

        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

        x_major_ticks = np.arange(x_min, x_max+0.01, 2*x_step)
        x_minor_ticks = np.arange(x_min, x_max+0.01, x_step)

        y_major_ticks = np.arange(y_min, y_max+0.001, 2*y_step)
        y_minor_ticks = np.arange(y_min, y_max+0.001, y_step)

        ax.set_xticks(x_major_ticks)
        ax.set_xticks(x_minor_ticks, minor=True)

        ax.set_yticks(y_major_ticks)
        ax.set_yticks(y_minor_ticks, minor=True)

        ax.grid(which='both')

    if save:
        if not exists(path):
            makedirs(path)
        plt.savefig(path + graph_name + '_prj_2d' + '.png')
        plt.close()
    else:
        plt.show()


def node_space_plot_2D_elipsoid(embedding, color_values, labels=None, means=None, covariances=None, grid=False, path=None):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    # color_iter = ['lightgreen', 'cyan', 'purple', 'red']
    # color_iter = ['red', 'cyan', 'purple', 'lightgreen']
    color_iter = itertools.cycle(['lightgreen', 'purple', 'red', 'cyan'])
    # color_iter = itertools.cycle(['lightgreen', 'purple', 'cyan', 'red'])
    # color_iter = itertools.cycle(['cyan', 'red', 'purple', 'lightgreen'])

    nodes_id = np.array(list(range(1, len(embedding)+1)))
    data = np.concatenate((embedding, nodes_id.reshape(len(nodes_id), 1)), axis=1)

    for node in data:
        ax.scatter(node[0], node[1], c=color_values[node[2]-1], marker='o', s=100)
        ax.text(node[0], node[1],  '%s' % (str(int(node[2]))), size=10)


    if (means is not None) and (covariances is not None):
        for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
            v, w = np.linalg.eigh(3*covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            transparency = 0.45
            if not np.any(labels == i):
                transparency = 0.0

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(transparency)
            ax.add_artist(ell)


    if grid:
        x_max, x_min = 2.5, -2.5
        y_max, y_min = -0,-2.5

        x_step = (x_max - x_min) / 4.0
        y_step = (y_max - y_min) / 4.0

        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

        x_major_ticks = np.arange(x_min, x_max+0.01, 2*x_step)
        x_minor_ticks = np.arange(x_min, x_max+0.01, x_step)

        y_major_ticks = np.arange(y_min, y_max+0.001, 2*y_step)
        y_minor_ticks = np.arange(y_min, y_max+0.001, y_step)

        ax.set_xticks(x_major_ticks)
        ax.set_xticks(x_minor_ticks, minor=True)

        ax.set_yticks(y_major_ticks)
        ax.set_yticks(y_minor_ticks, minor=True)

        ax.grid(which='both')

    if path:
        plt.savefig(path + '.png')
        plt.close()
    else:
        plt.show()