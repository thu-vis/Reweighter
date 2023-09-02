"""
sparse binary matrix
"""
# pylint: disable=invalid-name

from cmath import nan
from collections import Counter, defaultdict

import numpy as np
from numpy import ceil, exp, isfinite, log2


def entropy_bits(ary):
    """ -log2 of ary, defining -log2(0) ~approx Inf """
    ary += exp(-700)
    return -log2(ary)


def logstar(num):
    """ log-star (universal integer code length) of num """
    ell = 0
    while num > 1:
        ell += 1
        num = log2(num)
    return ell


def int_bits(ary):
    """ log2 of A, defining log2(0) = 0 and rounding up """
    ary[ary == 0] = 1
    return ceil(log2(ary))


def _rc_cluster_cost(ary):
    """ cluster cost for row and column groupings helper """
    dim = len(ary)
    cst = np.cumsum(ary)
    cst = cst[::-1] - dim + range(1, dim + 1)
    return sum(int_bits(cst))


class TriMatrix:
    """ spare matrix and all the parameters """

    def __init__(self, matrix_object, coef=3):
        # the _matrix object is assumed to be a csc matrix
        self._matrix = matrix_object
        self.coef = coef
        num_rows, num_cols = self._matrix.shape
        self._row_clusters = [0] * num_rows
        self._col_clusters = [0] * num_cols
        self._col_cluster_sizes = [0]
        self._row_cluster_sizes = [0]
        self._D_pos = None
        self._D_neg = None
        self._entropy_terms = None
        self._cached_max_row_cluster = max(self._row_clusters)
        self._cached_max_col_cluster = max(self._col_clusters)
        self._dok_copy = matrix_object.asformat("dok")
        self._csr_copy = matrix_object.asformat("csr")
        self.build_row_deltas()
        self.build_col_deltas()

    def kmeans_init(self, nrow, ncol):
        from sklearn.cluster import KMeans
        X = self._matrix.toarray()
        kmeans = KMeans(n_clusters=nrow, random_state=0).fit(X)
        self.set_row_cluster(kmeans.labels_)
        kmeans = KMeans(n_clusters=ncol, random_state=0).fit(X.T)
        self.set_col_cluster(kmeans.labels_)

    def set_row_cluster(self, ary):
        assert len(self._row_clusters) == len(ary)
        self._row_clusters = ary
        self._cached_max_row_cluster = max(self._row_clusters)
        self._D_pos = None
        self._D_neg = None
        c = Counter(self._row_clusters)
        self._row_cluster_sizes = [0] * (self._cached_max_row_cluster + 1)
        for cidx in c:
            self._row_cluster_sizes[cidx] += c[cidx]
        self.build_row_deltas()
    
    def set_col_cluster(self, ary):
        assert len(self._col_clusters) == len(ary)
        self._col_clusters = ary
        self._cached_max_col_cluster = max(self._col_clusters)
        self._D_pos = None
        self._D_neg = None
        c = Counter(self._col_clusters)
        self._col_cluster_sizes = [0] * (self._cached_max_col_cluster + 1)
        for cidx in c:
            self._col_cluster_sizes[cidx] += c[cidx]
        self.build_col_deltas()

    def copy(self):
        return (self._D_pos.copy(), self._D_neg.copy(), self._entropy_terms.copy(),\
        self._cached_max_row_cluster, self._cached_max_col_cluster,\
        self._col_clusters.copy(), self._row_clusters.copy(), \
        self._row_cluster_sizes.copy(), self._col_cluster_sizes.copy())

    def update(self, storage):
        self._D_pos, self._D_neg, self._entropy_terms,\
        self._cached_max_row_cluster, self._cached_max_col_cluster,\
        self._col_clusters, self._row_clusters, self._row_cluster_sizes,\
        self._col_cluster_sizes = storage

    @property
    def matrix(self):
        """
        Return the sparse matrix object
        """
        return self._matrix

    @property
    def col_clusters(self):
        """
        Return array of column cluster assignments
        """
        return self._col_clusters

    @property
    def row_clusters(self):
        """
        Return array of row cluster assignments
        """
        return self._row_clusters

    @staticmethod
    def _resequence(cluster_labels):
        """
        Yield a squashed version of the cluster indexes.
        :param cluster_labels: the cluster index values
        """
        mp = {b: a for a, b in enumerate(sorted(np.unique(cluster_labels)))}
        for elem in cluster_labels:
            yield mp[elem]

    def defrag_clusters(self):
        """
        Remove empty clusters
        """
        return 
        self._row_clusters = np.array([_ for _ in self._resequence(self._row_clusters)])
        self._col_clusters = np.array([_ for _ in self._resequence(self._col_clusters)])
        self._D_pos = None
        self._D_neg = None
        self._cached_max_col_cluster = max(self._col_clusters)
        self._cached_max_row_cluster = max(self._row_clusters)

    @property
    def num_row_clusters(self):
        """
        Return the number of row clusters
        """
        return len(self._row_cluster_sizes)

    @property
    def num_col_clusters(self):
        """
        Return the number of column clusters
        """
        return len(self._col_cluster_sizes)

    def recalc_D(self):
        """
        Return the non-zero counts of the cross associations.

        This is a heavy computation, so we cache and update as much as
        possible
        """
        self._D_pos = np.zeros(
            (self.num_row_clusters, self.num_col_clusters), dtype=np.uint32)
        self._D_neg = np.zeros(
            (self.num_row_clusters, self.num_col_clusters), dtype=np.uint32)
        rDict, cDict = dict(), dict()
        R, C = np.array(self._row_clusters), np.array(self._col_clusters)
        for i in range(self.num_row_clusters): rDict[i] = np.where(R == i)[0]
        for j in range(self.num_col_clusters): cDict[j] = np.where(C == j)[0]
        for i in range(self.num_row_clusters):
            for j in range(self.num_col_clusters):
                tmp = self._dok_copy[rDict[i], :][:, cDict[j]]
                self._D_pos[i, j] = np.sum(tmp==1)
                self._D_neg[i, j] = self._D_pos[i, j] - tmp.sum()
        return 

    @property
    def co_cluster_sizes(self):
        """
        Return the co-cluster sizes (Nxy)
        """
        return np.outer(
            self._row_cluster_sizes,
            self._col_cluster_sizes
        )

    def _update_entropy_terms(self):
        Dp, Dn = self._D_pos, self._D_neg
        Nxy = self.co_cluster_sizes
        if self._D_pos is None or Dp.shape != Nxy.shape:
            self.recalc_D()
            Dp, Dn = self._D_pos, self._D_neg
        Dz = Nxy - Dp - Dn
        Pz = Dz / Nxy; Pz[~isfinite(Pz)] = 0
        Pp = Dp / Nxy; Pp[~isfinite(Pp)] = 0
        Pn = Dn / Nxy; Pn[~isfinite(Pn)] = 0
        # self._entropy_terms = np.multiply(Dz, entropy_bits(Pz)) + np.multiply(Dp, entropy_bits(Pp)) + np.multiply(Dn, entropy_bits(Pn))
        # self._entropy_terms = np.multiply(Dp, entropy_bits(Pp)) + np.multiply(Dn, entropy_bits(Pn))
        self._entropy_terms = np.multiply(Dz, entropy_bits(Pz)) + np.multiply(Dp, entropy_bits(Pp)) + np.multiply(Dn, entropy_bits(Pn))
        
    
    @property
    def per_block_cost(self):
        return self._entropy_terms.sum()


    @property
    def cost(self):
        """
        returns:(total encoding cost, per-block 0/1s only)
        """
        # col and row
        # cost = logstar(self.num_col_clusters) + \
        #     logstar(self.num_row_clusters)
        # cluster sizes
        # cost += self._cluster_size_cost
        # cost += int_bits(self.co_cluster_sizes + 1).sum()
        # num_rows, num_cols = self._matrix.shape
        # cost = num_rows * ceil(log2(self.num_row_clusters)) + num_cols * ceil(log2(ary))
        # block cost
        cost = self.coef * self.num_row_clusters * self.num_col_clusters
        self._update_entropy_terms()
        cost += self.per_block_cost
        return cost, self.per_block_cost

    @property
    def _cluster_size_cost(self):
        """
        Compute the cost of the cluster sizes
        """
        return _rc_cluster_cost(self._row_cluster_sizes) + \
            _rc_cluster_cost(self._col_cluster_sizes)

    def _cluster_entropies(self, orientation="row"):
        """
        Compute the row or column block entropies
        :param orientation: row or col
        """
        if orientation == 'col':
            return np.sum(self._entropy_terms, axis=0) / self._col_cluster_sizes
        elif orientation == 'row':
            return np.sum(self._entropy_terms, axis=1) / self._row_cluster_sizes

    def max_entropy_cluster(self, orientation="row"):
        """
        Return the value and index of the worst cluster for the given
        orientation

        :param orientation: row or column
        """
        entropies = self._cluster_entropies(orientation=orientation)
        idx = np.argmax(entropies)
        val = entropies[idx]
        return idx, val

    def entropy_of_cluster(self, index, orientation="row"):
        """
        :param index: index of the cluster to check
        :param orientation: row or col
        """
        return self._cluster_entropies(orientation=orientation)[index]

    def cluster_members(self, index, orientation="row"):
        """
        return the index values that belong to the cluster

        :param index: the index value
        :param orientation: row or column
        """
        if orientation == 'row':
            Q = self._row_clusters
        elif orientation == 'col':
            Q = self._col_clusters
        return np.where(Q==index)[0].tolist()

    def _get_row(self, index):
        row_values_coo = self._csr_copy[index].tocoo()
        return zip(row_values_coo.col, row_values_coo.data)

    def _get_col(self, index):
        col_values_coo = self._csr_copy[:, index].tocoo()
        return zip(col_values_coo.row, col_values_coo.data)

    def hstack_dnz(self):
        """
        Add a column for new counts
        """
        extra_cluster = [[0] for _ in range(self._D_pos.shape[0])]
        self._D_pos = np.hstack((self._D_pos, extra_cluster))
        self._D_neg = np.hstack((self._D_neg, extra_cluster))
        self._entropy_terms = np.hstack((self._entropy_terms, extra_cluster))
        self._cached_max_col_cluster += 1
        self._col_cluster_sizes.append(0)

    def vstack_dnz(self):
        """
        Add a row for new counts
        """
        extra_cluster = [0 for _ in range(self._D_pos.shape[1])]
        self._D_pos = np.vstack((self._D_pos, extra_cluster))
        self._D_neg = np.vstack((self._D_neg, extra_cluster))
        self._entropy_terms = np.vstack((self._entropy_terms, extra_cluster))
        self._cached_max_row_cluster += 1
        self._row_cluster_sizes.append(0)

    @property
    def transformed_matrix(self):
        """
        return seriated matrix
        """
        cx = Counter(self.col_clusters)
        idx_x = []
        for cluster_id, _ in cx.most_common():
            idx_x.append(self.cluster_members(cluster_id, orientation="col"))
        cy = Counter(self.row_clusters)
        idx_y = []
        for cluster_id, _ in cy.most_common():
            idx_y.append(self.cluster_members(cluster_id, orientation="row"))
        # this index trick may depend on the type
        idx_x = [_ for ary in idx_x for _ in ary]
        idx_y = [_ for ary in idx_y for _ in ary]
        return self._matrix[:, idx_x][idx_y]

    def find_best_col_cluster(self, index, cluster_ids):
        """
        Update a cluster index
        :param index: the index to update
        :param new_col_cluster: alternative clusters
        """
        old_col_cluster = self._col_clusters[index]
        dp_deltas, dn_deltas = self.row_deltas_dict[index]

        self._D_pos[:, old_col_cluster] -= dp_deltas
        self._D_neg[:, old_col_cluster] -= dn_deltas
        self._col_cluster_sizes[old_col_cluster] -= 1
        self._update_entropy_terms()
        base_entropy = np.sum(self._entropy_terms, axis=0)

        self._D_pos[:, cluster_ids] += dp_deltas.reshape(-1, 1)
        self._D_neg[:, cluster_ids] += dn_deltas.reshape(-1, 1)
        for idx in cluster_ids:
            self._col_cluster_sizes[idx] += 1

        self._update_entropy_terms()
        new_entropy = np.sum(self._entropy_terms, axis=0)
        new_col_cluster = np.argmin(new_entropy - base_entropy)

        self._col_clusters[index] = new_col_cluster
        self._D_pos[:, cluster_ids] -= dp_deltas.reshape(-1, 1)
        self._D_neg[:, cluster_ids] -= dn_deltas.reshape(-1, 1)
        self._D_pos[:, new_col_cluster] += dp_deltas
        self._D_neg[:, new_col_cluster] += dn_deltas
        for idx in cluster_ids:
            self._col_cluster_sizes[idx] -= 1
        self._col_cluster_sizes[new_col_cluster] += 1

        self._update_entropy_terms()


    def find_best_row_cluster(self, index, cluster_ids):
        """
        Update a cluster index
        :param index: the index to update
        :param new_row_cluster: the new row cluster
        """
        # vica: find the best new_row_cluster here, we dont need to update total cost 
        old_row_cluster = self._row_clusters[index]

        dp_deltas, dn_deltas = self.col_deltas_dict[index]
        self._D_pos[old_row_cluster] -= dp_deltas
        self._D_neg[old_row_cluster] -= dn_deltas
        self._row_cluster_sizes[old_row_cluster] -= 1
        self._update_entropy_terms()
        base_entropy = np.sum(self._entropy_terms, axis=1)

        self._D_pos[cluster_ids] += dp_deltas
        self._D_neg[cluster_ids] += dn_deltas
        for idx in cluster_ids:
            self._row_cluster_sizes[idx] += 1
        self._update_entropy_terms()

        new_entropy = np.sum(self._entropy_terms, axis=1)
        new_row_cluster = np.argmin(new_entropy - base_entropy)
        self._row_clusters[index] = new_row_cluster
        self._D_pos[cluster_ids] -= dp_deltas
        self._D_neg[cluster_ids] -= dn_deltas
        self._D_pos[new_row_cluster] += dp_deltas
        self._D_neg[new_row_cluster] += dn_deltas

        for idx in cluster_ids:
            self._row_cluster_sizes[idx] -= 1
        self._row_cluster_sizes[new_row_cluster] += 1
        self._update_entropy_terms()

    def update_col_cluster(self, index, new_col_cluster):
        """
        Update a cluster index
        :param index: the index to update
        :param new_col_cluster: the new col cluster
        """
        old_col_cluster = self._col_clusters[index]
        if old_col_cluster == new_col_cluster:
            return
        dp_deltas, dn_deltas = self.row_deltas_dict[index]
        self._D_pos[:, old_col_cluster] -= dp_deltas
        self._D_pos[:, new_col_cluster] += dp_deltas
        self._D_neg[:, old_col_cluster] -= dn_deltas
        self._D_neg[:, new_col_cluster] += dn_deltas
        
        self._col_clusters[index] = new_col_cluster
        self._col_cluster_sizes[old_col_cluster] -= 1
        self._col_cluster_sizes[new_col_cluster] += 1
        self._update_entropy_terms()


    def update_row_cluster(self, index, new_row_cluster):
        """
        Update a cluster index
        :param index: the index to update
        :param new_row_cluster: the new row cluster
        """
        # vica: find the best new_row_cluster here, we dont need to update total cost 
        old_row_cluster = self._row_clusters[index]
        if old_row_cluster == new_row_cluster:
            return
        dp_deltas, dn_deltas = self.col_deltas_dict[index]

        self._D_pos[old_row_cluster] -= dp_deltas
        self._D_pos[new_row_cluster] += dp_deltas
        self._D_neg[old_row_cluster] -= dn_deltas
        self._D_neg[new_row_cluster] += dn_deltas
        
        self._row_clusters[index] = new_row_cluster
        self._row_cluster_sizes[old_row_cluster] -= 1
        self._row_cluster_sizes[new_row_cluster] += 1
        self._update_entropy_terms()


    def build_row_deltas(self):
        self.row_deltas_dict = []
        for index, _ in enumerate(self._col_clusters):
            col_values = self._get_col(index)
            dp_deltas = np.zeros(len(self._row_cluster_sizes), dtype="int64")
            dn_deltas = np.zeros(len(self._row_cluster_sizes), dtype="int64")
            for row, data in col_values:
                if data == 1: dp_deltas[self._row_clusters[row]] += 1
                if data == -1: dn_deltas[self._row_clusters[row]] += 1

            self.row_deltas_dict.append((dp_deltas, dn_deltas))

    def build_col_deltas(self):
        self.col_deltas_dict = []
        for index, _ in enumerate(self._row_clusters):
            row_values = self._get_row(index)
            dp_deltas = np.zeros(len(self._col_cluster_sizes), dtype="int64")
            dn_deltas = np.zeros(len(self._col_cluster_sizes), dtype="int64")
            for col, data in row_values:
                if data == 1: dp_deltas[self._col_clusters[col]] += 1
                if data == -1: dn_deltas[self._col_clusters[col]] += 1
    
            self.col_deltas_dict.append((dp_deltas, dn_deltas))