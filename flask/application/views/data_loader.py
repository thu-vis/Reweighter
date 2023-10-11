
import os
import os.path as osp
from application.views.utils.config_utils import config
from application.views.utils.helper_utils import json_load_data
from glob import glob
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from application.views.LP import LinearProgramming
from application.views.coclustering import CoClustering
from application.views.FACA import search_col, search_row, search
import warnings
import scipy.sparse

class DataLoader:

    def __init__(self, name):
        self.name = name
        self.data_root = osp.join(config.root, 'data', name)
        self.cache_root = osp.join(self.data_root, 'cache')
        os.makedirs(self.cache_root, exist_ok=True)
        self.encoded_images = json_load_data(osp.join(self.data_root, 'encoded_images.json'))
        self.gts = np.load(osp.join(self.data_root, 'gts.npy')) # y_gold in learning
        self.labels = np.load(osp.join(self.data_root, 'labels.npy')) # y_train in learning
        if osp.exists(osp.join(self.data_root, 'extra_R.npz')): 
            self.extra_R = np.load(osp.join(self.data_root, 'extra_R.npz'))['R']
            self.extra_R_row_index = list(np.load(osp.join(self.data_root, 'extra_R.npz'))['row_index'])
        else:
            self.extra_R = None
            self.extra_R_row_index = None
        if osp.exists(osp.join(self.data_root, 'add_R.npz')): 
            self.add_R_data = np.load(osp.join(self.data_root, 'add_R.npz'))['R']
            self.add_R_row_index = list(np.load(osp.join(self.data_root, 'add_R.npz'))['row_index'])
        else:
            self.add_R_data = None
            self.add_R_row_index = None
        self.corrects = (self.gts == self.labels) # y_same in learning
        self.pred = np.array(self.labels)
        self.n_class = len(set(self.gts))
        self.label_names = json_load_data(osp.join(self.data_root, 'label_names.json'), 'utf-8')

        data = np.load(osp.join(self.data_root, f'visdata.npz'))
        self.R, self.row_index, self.col_index, self.pos_idx, self.neg_idx, self.display_idx = data['R'], data['row_index'], data['col_index'], data['pos_i'], data['neg_i'], data['display_i']
        self.row_label = self.labels[self.row_index]
        self.full_R = np.load(osp.join(self.data_root, 'full_R.npy'))
        self.pos_index_set = set(self.col_index[self.pos_idx])
        self.neg_index_set = set(self.col_index[self.neg_idx])
        import pickle
        with open(osp.join(self.data_root, 'hierachy.pkl'), 'rb') as f:
            self.full_col_cluster = pickle.load(f)
        self.full_col_pos = np.load(osp.join(self.data_root, 'full_coord.npz'))['col_pos']
        self.lam = np.ones(self.R.shape[0]) / self.R.shape[0]
        self.lam_factor = 1
        self.lam_lower, self.lam_upper = np.ones(self.R.shape[0])*1e-4, np.ones(self.R.shape[0])
        

        self.lower, self.upper = np.quantile(self.R, 0.05), np.quantile(self.R, 0.95)
        bin_m = np.zeros_like(self.R, dtype=np.int32)
        bin_m[self.R<self.lower] = -1
        bin_m[self.R>self.upper] = 1
        self.sparse_matrix = scipy.sparse.csc_matrix(bin_m)

        self.R = self.R 
        if self.extra_R is not None: self.extra_R = self.extra_R
        if self.add_R_data is not None: self.add_R_data = self.add_R_data

        self.all_index = np.array(list(set(self.row_index)|set(self.col_index)))


        self.weight, self.clip_weight, self.full_weight = self.calculate_weight()
        self.up_idx = np.array([])
        self.down_idx = np.array([])
        self.up_target = np.array([])
        self.down_target = np.array([])

        self.projection_method = 'TSNE'
        self.gamma = 0.1 
        

    def get_constraint(self):
        return {'pos_idx': self.pos_idx, 'neg_idx': self.neg_idx, 'lam_lower': self.lam_lower, 'lam_upper': self.lam_upper,
        'up_idx': self.up_idx, 'down_idx': self.down_idx, 'up_target': self.up_target, 'down_target': self.down_target}

    def set_constraint(self, data):
        import json
        with open('/data/vica/IEG/data.json', 'w') as f:
            json.dump(data, f)
        gamma = self.gamma
        self.pos_idx = np.array(data['pos_idx'], dtype=np.int32)
        self.neg_idx = np.array(data['neg_idx'], dtype=np.int32)
        for idx in data['reduce_idx']:
            self.lam_upper[idx] = self.lam[idx] * self.lam_factor * (1 - gamma)
        for idx in data['enlarge_idx']:
            self.lam_lower[idx] = self.lam[idx] * self.lam_factor * (1 + gamma)
        self.up_idx = np.concatenate([self.up_idx, np.array(data['up_idx'], dtype=np.int32)])
        self.down_idx = np.concatenate([self.down_idx, np.array(data['down_idx'], dtype=np.int32)])


    def get_cluster(self):
        data = np.load(osp.join(self.cache_root, f'FACA-{self.name}.npz'))
        row_cluster_label = data['row_cluster_label']
        col_cluster_label = data['col_cluster_label']
        self.row_cluster_num = len(set(row_cluster_label))
        self.col_cluster_num = len(set(col_cluster_label))
        coord = np.load(osp.join(self.cache_root, f'coord.npz'))
        self.row_pos = coord['row_pos']#.item()
        self.col_pos = coord['col_pos']#.item()

        self.row_cluster_label, self.col_cluster_label, col_map = self.reorder(row_cluster_label, col_cluster_label)
        for cluster in self.full_col_cluster:
            cluster['id'] = str(col_map[int(cluster['id'])])
        # self.row_cluster_label, self.col_cluster_label = row_cluster_label, col_cluster_label
        self.row_sub_cluster_label = np.zeros(self.row_cluster_label.shape, np.int) - 1
        self.col_sub_cluster_label = np.zeros(self.col_cluster_label.shape, np.int) - 1
        return {
            'R': self.R,
            'row_index': self.row_index,
            'row_x': self.row_pos.reshape(-1),
            'row_cluster_label': self.row_cluster_label, 
            'row_sub_cluster_label': self.row_sub_cluster_label,
            'col_index': self.col_index, 
            'col_cluster_label': self.col_cluster_label,
            'col_sub_cluster_label': self.col_sub_cluster_label,
            'col_x': self.col_pos.reshape(-1),
            'all_index': self.all_index,
            'display_idx': self.display_idx,
            'num_row_cluster': self.row_cluster_num,
            'num_col_cluster': self.col_cluster_num,
            'full_col_cluster': self.full_col_cluster,
        }
    

    @staticmethod
    def reorder_by_score(cluster_labels, scores):
        unique_labels = np.unique(cluster_labels)
        scores = [scores[np.where(cluster_labels==l)[0]].mean() for l in unique_labels]
        label_map = {unique_labels[i]: new_l for (new_l, i) in enumerate(np.argsort(scores))}
        return np.array([label_map[l] for l in cluster_labels]), label_map

    def reorder(self, row_cluster_label, col_cluster_label):
        new_row_cluster_label, _ = DataLoader.reorder_by_score(row_cluster_label, self.labels[self.row_index])
        new_col_cluster_label, col_map = DataLoader.reorder_by_score(col_cluster_label, self.labels[self.col_index])
        return new_row_cluster_label, new_col_cluster_label, col_map

    def projection(self, features, n_dimensions=1, method='TSNE'):
        assert len(features.shape) == 2
        if features.shape[0] == 1:
            return np.array([0.5])
        if features.shape[0] == 2:
            return np.array([0,1])
        perplexity = np.sqrt(features.shape[0])
        with warnings.catch_warnings():
            # ignore all caught warnings
            warnings.filterwarnings("ignore")
            if method == 'TSNE': projector = TSNE(n_components=n_dimensions, init='pca', learning_rate=500, random_state=0, perplexity=perplexity)
            elif method == 'PCA': projector = PCA(n_components=n_dimensions, random_state=0)
            else: raise NotImplementedError('unknown projection method: ', method)
            X = projector.fit_transform(features)
            return X

    def calculate_weight(self):
        lam = self.lam / (self.lam.sum()+1e-5)
        weight = np.matmul(lam, self.R)
        clip_weight = np.clip(weight, a_min=0, a_max=None)
        clip_weight = clip_weight / (clip_weight.sum()+1e-5)
        full_weight = np.matmul(lam, self.full_R)
        full_weight[self.col_index] = weight
        return weight, clip_weight, full_weight

    def update_weight(self):
        self.old_lam, self.old_weight, self.old_clip_weight, self.old_full_weight = self.lam, self.weight, self.clip_weight, self.full_weight
        R = self.R
        lam = np.ones(self.R.shape[0]) / self.n_class
        for l in range(self.n_class):
            lam[self.row_label==l] /= (self.row_label==l).sum()
        used_i = np.concatenate([self.pos_idx, self.neg_idx])
        used_col_index = self.col_index[used_i]
        optimizer = LinearProgramming(self.R[:, used_i], self.corrects[used_col_index], lam, self.labels[used_col_index], np.arange(len(self.pos_idx)), np.arange(len(self.pos_idx)+len(self.neg_idx)), (1,1,1), 1)
        optimizer.set_lam_range(self.lam_lower, self.lam_upper)
        optimizer.set_active_goal(np.concatenate((self.up_idx, self.down_idx)), np.concatenate((np.ones(len(self.up_idx)), np.zeros(len(self.down_idx)))))

        optimizer.optimize(.1, freq=1000)
        self.lam = optimizer.lam.detach().numpy()
        self.lam_factor = self.lam.sum()
        self.lam = (self.lam / (self.lam.sum()+1e-5))
        self.weight, self.clip_weight, self.full_weight = self.calculate_weight()
        return self.old_lam, self.old_weight, self.old_clip_weight, self.old_full_weight, self.lam, self.weight, self.clip_weight, self.full_weight

    def set_sub_cluster_label(self, type, labels, local_idx):
        if type == 'row':
            self.row_sub_cluster_label[local_idx] = labels
        elif type == 'col':
            self.col_sub_cluster_label[local_idx] = labels
        else:
            raise NotImplementedError('unknown type: ', type)

    def replace_R(self, row_index, row_label):
        # currently I ignore the used of label, i.e., the users always choose correct labels
        # if we want to support arbitrary label, we need to cache more, or dynamically calculate it
        # update R
        changed_row_cluster = []
        previous_row_cluster_label = np.array(self.row_cluster_label)
        for i, l in zip(row_index, row_label):
            local_i = list(self.row_index).index(i)
            local_i2 = self.extra_R_row_index.index(i)
            previous_row_cluster_label[local_i] = -1
            self.R[local_i,:] = self.extra_R[local_i2,:]
            # self.full_R[,:] = self.extra_R[local_i2,:]
            self.row_label[local_i] = l
            self.lam_lower[local_i] = 1 / self.R.shape[0] * self.lam_factor / 2
        # update validation cluster info
        means = np.zeros((self.row_cluster_num, self.R.shape[1]))
        for l in range(self.row_cluster_num):
            sub_row_index = (previous_row_cluster_label == l)
            if sub_row_index.sum() > 0:
                means[l] = self.R[sub_row_index, :].mean(axis=0)
        for i in row_index:
            local_i = list(self.row_index).index(i)
            dists = ((self.R[local_i,] - means)**2).sum(axis=1)
            l = dists.argmin()
            previous_row_cluster_label[local_i] = l
            changed_row_cluster.append(l)
            changed_row_cluster.append(self.row_cluster_label[local_i])
        self.row_cluster_label = previous_row_cluster_label
        changed_row_cluster = set(changed_row_cluster)
        cluster_info = dict()
        for l in changed_row_cluster:
            cluster_pos = np.where(self.row_cluster_label == l)[0]
            cluster_X = self.row_pos[cluster_pos]
            cluster_info[str(l)] = {
                'index': self.row_index[cluster_pos],
                'local_i': cluster_pos,
                'row_x': cluster_X,
                'cluster_label': l,
            }

        ret = {'R': self.R, 'cluster_info': cluster_info, 'rows': list(map(str, changed_row_cluster))}
        return ret

    def add_R(self, add_row_index, add_row_label):
        new_R = self.add_R_data[[self.add_R_row_index.index(i) for i in add_row_index],:]

        keep_col = np.ones(len(self.col_index), dtype=np.bool)
        for idx in add_row_index: keep_col[list(self.col_index).index(idx)] = False
        old_R, old_nrow, old_ncol = self.R, self.R.shape[0], self.R.shape[1]
        n_new = len(add_row_index)
        self.R = np.zeros((old_nrow + n_new, old_ncol - n_new))
        self.R[:old_nrow, :] = old_R[:, keep_col]
        self.R[old_nrow:, :] = new_R[:, keep_col]

        bin_m = np.zeros_like(self.R, dtype=np.int32)
        bin_m[self.R<self.lower] = -1
        bin_m[self.R>self.upper] = 1
        self.sparse_matrix = scipy.sparse.csc_matrix(bin_m)

        self.row_cluster_label = np.concatenate([self.row_cluster_label, np.ones(n_new, np.int32) - 1])
        means = np.zeros((old_nrow, self.R.shape[1]))
        for l in range(self.row_cluster_num):
            sub_row_index = (self.row_cluster_label == l)
            if sub_row_index.sum() > 0:
                means[l] = self.R[sub_row_index, :].mean(axis=0)

        changed_row_cluster = []
        for row in range(old_nrow, old_nrow+n_new):
            dists = ((self.R[row,] - means)**2).sum(axis=1)
            l = dists.argmin()
            self.row_cluster_label[row] = l
            changed_row_cluster.append(l)
        changed_row_cluster = list(set(changed_row_cluster))

        self.col_cluster_label = self.col_cluster_label[keep_col]

        self.row_index = np.concatenate([self.row_index, add_row_index])
        self.col_index = self.col_index[keep_col]
        self.row_label = self.labels[self.row_index]
        
        feat_row_pos = np.zeros(self.row_index.shape[0])
        feat_row_pos[:old_nrow] = self.row_pos

        cluster_info = dict()
        for l in changed_row_cluster:
            cluster_pos = np.where(self.row_cluster_label == l)[0]
            cluster_X = self.projection(self.sparse_matrix[cluster_pos,:].toarray(), 1, self.projection_method).reshape(-1)
            feat_row_pos[cluster_pos] = cluster_X
            cluster_info[str(l)] = {
                'index': self.row_index[cluster_pos],
                'local_i': cluster_pos,
                'row_x': cluster_X,
                'cluster_label': l,
                # 'sub_cluster_label': sub_cluster,
            }
        self.row_pos = feat_row_pos
        self.col_pos = self.col_pos[keep_col]

        # update constraint related
        previous_constraints = [dict() for i in range(old_ncol)]
        for i in self.pos_idx: previous_constraints[i]['pos'] = True
        for i in self.neg_idx: previous_constraints[i]['neg'] = True
        for i, val in zip(self.up_idx, self.up_target): previous_constraints[i]['up'] = val
        for i, val in zip(self.down_idx, self.down_target): previous_constraints[i]['down'] = val
        previous_constraints = [previous_constraints[i] for i in range(old_ncol) if keep_col[i]]
        self.pos_idx = np.array([i for i, data in enumerate(previous_constraints) if 'pos' in data])
        self.neg_idx = np.array([i for i, data in enumerate(previous_constraints) if 'neg' in data])
        self.up_idx = np.array([i for i, data in enumerate(previous_constraints) if 'up' in data])
        self.up_target = np.array([previous_constraints[i]['up'] for i in self.up_idx])
        self.down_idx = np.array([i for i, data in enumerate(previous_constraints) if 'down' in data])
        self.down_target = np.array([previous_constraints[i]['down'] for i in self.down_idx])
        self.lam_lower = np.concatenate([self.lam_lower, np.ones(n_new)*1e-4])
        self.lam_upper = np.concatenate([self.lam_upper, np.ones(n_new)])

        # update weight and lam
        ratio = 2 * n_new / (old_nrow + n_new)
        self.lam = np.concatenate([self.lam * (1-ratio), np.ones(n_new) / n_new * ratio])
        self.weight = self.weight[keep_col]
        self.clip_weight = self.clip_weight[keep_col]

        constraint = self.get_constraint()
        ret = {'R': self.R, 
        'cluster_info': cluster_info, 'rows': list(map(str, changed_row_cluster)),
        'row_index': self.row_index, 'col_index': self.col_index, 'add_row_index': add_row_index,
        'constraint': constraint}
        return ret


    def get_sub_cluster(self, type, local_idx, n):
        if type == 'row':
            sub_matrix = self.sparse_matrix[local_idx, :]
            labels = search_row(sub_matrix, self.col_cluster_label, n)[0]
        elif type == 'col':
            sub_matrix = self.sparse_matrix[:, local_idx]
            labels = search_col(sub_matrix, self.row_cluster_label, n)[1]

        scores = [self.row_pos[i] for i in local_idx] if type == 'row' else [self.col_pos[i] for i in local_idx]
        labels = DataLoader.reorder_by_score(labels, np.array(scores))
        self.set_sub_cluster_label(type, labels, local_idx)
        return labels

    def set_gamma(self, gamma):
        self.gamma = gamma
