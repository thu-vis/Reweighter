"""
perform a cluster search
"""
import logging
import random
import sys
from collections import namedtuple
from math import inf

import numpy as np
import scipy.io

from application.views.trimatrix import TriMatrix
# from cross_associations.plots import plot_shaded

import os

EPSILON = 1e-05


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CostTracker:
    """
    Convenience cost tracking
    """

    def __init__(self):
        self._cost_history = [(0, 0), (0, 0)]

    def track(self, cost):
        """
        Add an observation to the history
        :param cost: tuple of total and block costs
        """
        self._cost_history.append(cost)

    @property
    def current_cost_str(self):
        """
        Construct a human-readable string
        """
        current_cost = self._cost_history[-1]
        previous_cost = self._cost_history[-2]
        return "Global: {} ({}) Block-encoding: {} ({})".format(
            current_cost[0],
            current_cost[0] - previous_cost[0],
            current_cost[1],
            current_cost[1] - previous_cost[1]
        )

    @property
    def matlab_style_cost(self):
        """
        cost 8233 (C2: 8205)
        """
        return "cost {:.0f} (C2: {:.0f})".format(*self._cost_history[-1])

    @property
    def current_block_cost(self):
        """
        Get the current block encoding cost
        """
        return self._cost_history[-1][1]

    @property
    def current_total_cost(self):
        """
        Get the current total cost
        """
        return self._cost_history[-1][0]

    def global_improved(self):
        """
        Evaluate whether the global cost went down
        """
        return self._cost_history[-1][0] < self._cost_history[-2][0]

    def block_improved(self):
        """
        Evaluate whether the block cost went down
        """
        return self._cost_history[-1][1] < self._cost_history[-2][1]


class ClusterSearch:
    """
    Search for a good clustering: add a new cluster, reshape old clusters
    """

    def __init__(self, matrix_object, random_seed=42, figdir=None, signed=False):
        self._matrix = matrix_object
        self.figdir = figdir
        self.signed = signed
        if self.figdir: os.makedirs(self.figdir, exist_ok=True)
        self._num_row_clusters = 1
        self._num_col_clusters = 1
        self._random_seed = random_seed
        self._tracker = CostTracker()
        np.random.seed(self._random_seed)
        random.seed(self._random_seed)
        self.best_cost = inf
        self.best_so_far = None

    def track(self):
        """
        Add a tracking observation to the history
        """
        self._tracker.track(self._matrix.cost)

    def get_cost_string(self, style="col"):
        if style == 'col':
            fstring = '## (k={:d}) l={:d} Current {}'
        elif style == 'row':
            fstring = '## k={:d} (l={:d}) Current {}'
        elif style == 'final':
            fstring = '## k={:d} l={:d} Final {}'
        return fstring.format(
            self._matrix.num_row_clusters,
            self._matrix.num_col_clusters,
            self._tracker.matlab_style_cost
        )

    def update_best(self):
        if self._tracker.current_total_cost < self.best_cost:
            self.best_cost = self._tracker.current_total_cost
            self.best_so_far = (
                np.array(self._matrix.row_clusters[:]),
                np.array(self._matrix.col_clusters[:]))

    def get_clusters(self):
        return self._matrix.row_clusters, self._matrix.col_clusters

    def sub_run(self):
        """
        Run the clustering algorithm
        """
        self.track()
        logger.debug("## Starting {}".format(self._tracker.matlab_style_cost))
        while True:
            old_state = self._matrix.copy()
            old_cost = self._matrix.cost[0]

            # if(max(self._matrix._col_cluster_sizes) / min(self._matrix._col_cluster_sizes) < 10):
            if(min(self._matrix._col_cluster_sizes) > 0):
                self.add_col_cluster()
                self.reshape_col()
                col_state = self._matrix.copy()
                col_cost = self._matrix.cost[0]
            else:
                col_state = old_state
                col_cost = old_cost
            self._matrix.update(old_state)
            self._matrix.build_col_deltas()

            if(col_cost >= old_cost):
                self._matrix.update(old_state)
                self.track()
                break
            else:
                self._matrix.update(col_state)
                self._matrix.build_col_deltas()

            self.track()
            self.update_best()
        (self._matrix._row_clusters,
         self._matrix._col_clusters) = self.best_so_far
        logger.debug(self.get_cost_string(style="final"))
        # if self.figdir:
        #     plot_shaded(self._matrix, f'{self.figdir}/final.png', self.signed)
        #     np.savez(f'{self.figdir}/final.npz', col_cluster=self._matrix.col_clusters, row_cluster=self._matrix.row_clusters)

    def run(self):
        """
        Run the clustering algorithm
        """
        _axis = namedtuple("Axis", ("add_cluster", "reshape", "style"))
        col_axis = _axis(self.add_col_cluster, self.reshape_col, "col")
        row_axis = _axis(self.add_row_cluster, self.reshape_row, "row")
        self.track()
        logger.debug("## Starting {}".format(self._tracker.matlab_style_cost))
        while True:
            old_state = self._matrix.copy()
            old_cost = self._matrix.cost[0]
            if(max(self._matrix._col_cluster_sizes) / min(self._matrix._col_cluster_sizes) < 10):
            # if(min(self._matrix._col_cluster_sizes) > 0):
                col_axis.add_cluster()
                col_axis.reshape()
                col_state = self._matrix.copy()
                col_cost = self._matrix.cost[0]
            else:
                col_state = old_state
                col_cost = old_cost
            self._matrix.update(old_state)
            self._matrix.build_col_deltas()


            if(max(self._matrix._row_cluster_sizes) / min(self._matrix._row_cluster_sizes) < 10):
            # if(min(self._matrix._row_cluster_sizes) > 0):
                row_axis.add_cluster()
                row_axis.reshape()
                row_state = self._matrix.copy()
                row_cost = self._matrix.cost[0]
            else:
                row_state = old_state
                row_cost = old_cost
            accept_row = row_cost < old_cost and min(row_state[-2]) > 2
            accept_col = col_cost < old_cost and min(col_state[-1]) > 2
            if not (accept_row or accept_col):
                print('break')
                self._matrix.update(old_state)
                self._matrix.build_row_deltas()
                self.track()
                break
            if not accept_col and accept_row:
                self._matrix.update(row_state)
                self._matrix.build_row_deltas()
                logger.debug(self.get_cost_string(style=row_axis.style))
            elif not accept_row and accept_col:
                self._matrix.update(col_state)
                self._matrix.build_row_deltas()
                self._matrix.build_col_deltas()
                logger.debug(self.get_cost_string(style=col_axis.style))
            elif row_cost < col_cost:
                self._matrix.update(row_state)
                self._matrix.build_row_deltas()
                logger.debug(self.get_cost_string(style=row_axis.style))
            else:
                self._matrix.update(col_state)
                self._matrix.build_row_deltas()
                self._matrix.build_col_deltas()
                logger.debug(self.get_cost_string(style=col_axis.style))
            self.track()
            self.update_best()

        (self._matrix._row_clusters,
         self._matrix._col_clusters) = self.best_so_far
        logger.debug(self.get_cost_string(style="final"))
        # if self.figdir:
        #     plot_shaded(self._matrix, f'{self.figdir}/phase_1.png', self.signed)
        #     np.savez(f'{self.figdir}/phase_1.npz', col_cluster=self._matrix.col_clusters, row_cluster=self._matrix.row_clusters)


    def _reshape(self, orientation="row"):
        """
        Given a clustering, see if it can be improved.
        """
        if orientation == "row":
            cluster_labels = self._matrix.row_clusters
            update_function = self._matrix.find_best_row_cluster
            final_function = self._matrix.build_row_deltas
        elif orientation == "col":
            cluster_labels = self._matrix.col_clusters
            update_function = self._matrix.find_best_col_cluster
            final_function = self._matrix.build_col_deltas
        cluster_ids = list(range(max(cluster_labels) + 1))
        while True:
            start_cost = self._matrix.per_block_cost
            for idx, _ in enumerate(cluster_labels):
                update_function(idx, cluster_ids)

            # in practice, reshape won't ever increase start cost
            if self._matrix.per_block_cost >= start_cost:
                break

            logger.debug("Intermediate cost {:.0f} (C2: {:.0f})".format(
                *self._matrix.cost))
        final_function()

    def reshape_col(self):
        """
        reshape the column clusters
        """
        self._reshape(orientation="col")

    def reshape_row(self):
        """
        reshape the row clusters
        """
        self._reshape(orientation="row")

    def add_col_cluster(self):
        """
        Add a new column cluster
        """
        self._add_cluster(orientation="col")

    def add_row_cluster(self):
        """
        Add a new row cluster
        """
        self._add_cluster(orientation="row")

    def _add_cluster(self, orientation="row"):
        """
        Add a new row or column cluster
        """
        # do this while Dnz and Nxy will agree
        cluster_index, cluster_entropy = self._matrix.max_entropy_cluster(
            orientation=orientation)

        if orientation == "col":
            self._matrix.hstack_dnz()
            cluster_labels = self._matrix.col_clusters
            update_function = self._matrix.update_col_cluster
            final_function = self._matrix.build_col_deltas
        elif orientation == "row":
            self._matrix.vstack_dnz()
            cluster_labels = self._matrix.row_clusters
            update_function = self._matrix.update_row_cluster
            final_function = self._matrix.build_row_deltas

        new_cluster_index = max(cluster_labels) + 1
        for idx in self._matrix.cluster_members(
                cluster_index, orientation=orientation):
            original = cluster_labels[idx]
            update_function(idx, new_cluster_index)
            new_entropy = self._matrix.entropy_of_cluster(
                cluster_index, orientation=orientation)
            if new_entropy <= cluster_entropy - EPSILON:
                cluster_entropy = new_entropy
            else:
                update_function(idx, original)
        final_function()



def split_row(matrix, nrow):
    matrix = TriMatrix(matrix)
    matrix.kmeans_init(nrow, 1)
    cluster_search = ClusterSearch(matrix, signed=True)
    cluster_search.reshape_row()
    return cluster_search._matrix.row_clusters

def split_col(matrix, ncol):
    matrix = TriMatrix(matrix)
    matrix.kmeans_init(1, ncol)
    cluster_search = ClusterSearch(matrix, signed=True)
    cluster_search.reshape_col()
    return cluster_search._matrix.col_clusters

def search_col(matrix, row_cluster_label, ncol_max):
    X = matrix.toarray()
    matrix = TriMatrix(matrix)
    matrix.set_row_cluster(row_cluster_label)
    matrix.set_col_cluster(np.zeros(matrix._matrix.shape[1], dtype=np.int))
    cluster_search = ClusterSearch(matrix, signed=True)
    cluster_search.track()
    while True:
        print('add')
        cluster_search.add_col_cluster()
        cluster_search.reshape_col()
        cluster_search.track()
        cluster_search.update_best()
        if cluster_search._tracker.current_block_cost == 0:
            break
        if not cluster_search._tracker.global_improved():
            break
        if len(set(cluster_search._matrix.col_clusters)) >= ncol_max:
            break

    (cluster_search._matrix._row_clusters,
        cluster_search._matrix._col_clusters) = cluster_search.best_so_far
    return cluster_search._matrix._row_clusters, cluster_search._matrix._col_clusters

def new_search_col(matrix, row_cluster_label, ncol_max, coef=20): # vica
    X = matrix.toarray()
    matrix = TriMatrix(matrix)
    matrix.coef = coef
    matrix.set_row_cluster(row_cluster_label)
    matrix.set_col_cluster(np.zeros(matrix._matrix.shape[1], dtype=np.int))
    cluster_search = ClusterSearch(matrix, signed=True)
    cluster_search.track()
    while True:
        print('add')
        cluster_search.add_col_cluster()
        cluster_search.reshape_col()
        cluster_search.track()
        cluster_search.update_best()
        if cluster_search._tracker.current_block_cost == 0:
            break
        if not cluster_search._tracker.global_improved():
            break
        if len(set(cluster_search._matrix.col_clusters)) >= ncol_max:
            break

    (cluster_search._matrix._row_clusters,
        cluster_search._matrix._col_clusters) = cluster_search.best_so_far
    return cluster_search._matrix._row_clusters, cluster_search._matrix._col_clusters


def search_row(matrix, col_cluster_label, nrow_max):
    X = matrix.toarray()
    matrix = TriMatrix(matrix)
    matrix.set_col_cluster(col_cluster_label)
    matrix.set_row_cluster(np.zeros(matrix._matrix.shape[0], dtype=np.int))
    cluster_search = ClusterSearch(matrix, signed=True)
    cluster_search.track()
    while True:
        cluster_search.add_row_cluster()
        cluster_search.reshape_row()
        cluster_search.track()
        cluster_search.update_best()
        if cluster_search._tracker.current_block_cost == 0:
            break
        if not cluster_search._tracker.global_improved():
            break
        if len(set(cluster_search._matrix.row_clusters)) >= nrow_max:
            break

    (cluster_search._matrix._row_clusters,
        cluster_search._matrix._col_clusters) = cluster_search.best_so_far
    return cluster_search._matrix._row_clusters, cluster_search._matrix._col_clusters


def search(matrix, nrow, ncol, nrow_max, ncol_max):
    tri_matrix = TriMatrix(matrix)
    tri_matrix.kmeans_init(nrow, ncol)
    cluster_search = ClusterSearch(tri_matrix, figdir=False, signed=True)
    cluster_search.run()
    return cluster_search._matrix._row_clusters, cluster_search._matrix._col_clusters
    row_cluster_indexs, col_cluster_indexs = cluster_search.get_clusters()
    num_clusters = max(col_cluster_indexs) + 1
    col_clusters = [[] for _ in range(0, num_clusters)]

    for i in range(0, len(col_cluster_indexs)):
        col_clusters[col_cluster_indexs[i]].append(i)

    num_rows, num_cols = np.shape(matrix)
    cluster_cnt = 0
    cnt = 1
    for col_cluster in col_clusters:
        print("divide column cluster ", cnt)
        cnt += 1
        
        cluster_size = len(col_cluster)
        if(cluster_size >= num_cols / num_clusters):
            sub_col_cluster = [0] * cluster_size
            sub_matrix = matrix[:, col_cluster].copy()
            sub_search = ClusterSearch(TriMatrix(sub_matrix), figdir= figdir+"_sub", signed=True)
            sub_search._matrix.set_row_cluster(row_cluster_indexs)
            sub_search._matrix.set_col_cluster(sub_col_cluster)

            col_index_dict = {}
            for i in range(0, cluster_size):
                col_index_dict[i] = col_cluster[i]
            sub_search.run()
            _, now_col_cluster = sub_search.get_clusters()
            for i in range(0, len(now_col_cluster)):
                col_cluster_indexs[col_index_dict[i]] = cluster_cnt + now_col_cluster[i]
            cluster_cnt += max(now_col_cluster) + 1

    cluster_search._matrix.set_col_cluster(col_cluster_indexs)
    # plot_shaded(cluster_search._matrix, f'{figdir}/final.png', True)
    np.savez(f'{figdir}/final.npz',
        col_cluster=col_cluster_indexs, row_cluster=row_cluster_indexs)

    return 0

if __name__ == "__main__":
    sys.exit(main())
