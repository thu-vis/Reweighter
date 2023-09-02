from flask import Flask, request, render_template, jsonify, send_from_directory
from flask import Blueprint, request
from flask import send_file
import os
import os.path as osp
import base64
import numpy as np
from application.views.utils.config_utils import config
from application.views.data_loader import DataLoader
from application.views.FACA import split_row, split_col, search_row, search_col
from PIL import Image
from io import BytesIO
import warnings
import hashlib

api = Blueprint("api", __name__)

dataloader = None

@api.route("/favicon.ico")
def get_fav():
    filepath = os.path.join(config.root, "static")
    return send_from_directory(filepath, "favicon.ico")


@api.route('/data')
def get_data():
    global dataloader

    dataloader = DataLoader(config.dataname)

    meta_data = {
        'num_samples': len(dataloader.gts),
        'num_classes': dataloader.n_class,
        'label_names': dataloader.label_names,
    }

    cluster_data = dataloader.get_cluster()
    constraint = dataloader.get_constraint()
    old_lam, old_weight, old_clip_weight, old_full_weight, new_lam, new_weight, new_clip_weight, new_full_weight = dataloader.update_weight()
    print(dataloader.gts.shape, dataloader.labels.shape, dataloader.corrects.shape, dataloader.pred.shape, dataloader.full_col_pos.shape)
    samples = [{
        'index': int(index), 
        'image': '',
        'gt': int(dataloader.gts[index]),
        'label': int(dataloader.labels[index]),
        'correct': bool(dataloader.corrects[index]),
        'pred': int(dataloader.pred[index]),
        'col_pos': round(float(dataloader.full_col_pos[index]), 3),
    } for index in range(len(dataloader.gts))]
    for index in dataloader.all_index:
        samples[index]['image'] = dataloader.encoded_images[index],
    opt_result = {
        'new_lam': new_lam,
        'new_weight': new_weight,
        'new_clip_weight': new_clip_weight,
        'new_full_weight': new_full_weight,
        'old_lam': old_lam,
        'old_weight': old_weight,
        'old_clip_weight': old_clip_weight,
        'old_full_weight': old_full_weight,
    }
    ret = {
        'samples': samples,
        'meta_data': meta_data,
        'cluster': cluster_data,
        'constraint': constraint,
        'opt_result': opt_result,
        'full_col_cluster': dataloader.full_col_cluster,
    }
    return jsonify(ret)


@api.route('/update_weight', methods=['POST'])
def get_update_weight():
    dataloader.set_constraint(request.json)
    old_lam, old_weight, old_clip_weight, old_full_weight, new_lam, new_weight, new_clip_weight, new_full_weight = dataloader.update_weight()
    return jsonify({
        'new_lam': new_lam,
        'new_weight': new_weight,
        'new_clip_weight': new_clip_weight,
        'new_full_weight': new_full_weight,
        'old_lam': old_lam,
        'old_weight': old_weight,
        'old_clip_weight': old_clip_weight,
        'old_full_weight': old_full_weight,
    })


def hash_str(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()


@api.route('/save_constraint', methods=['POST'])
def save():
    from datetime import datetime
    corrected_idx = request.json['corrected']
    corrected_labels = [dataloader.gts[i] for i in corrected_idx]
    clean_constraint_buffer = np.unique(np.concatenate([dataloader.row_index, np.array(corrected_idx)]))
    noisy_constraint_buffer = np.unique(np.setdiff1d(dataloader.col_index[dataloader.neg_idx], corrected_idx))
    print(corrected_idx, len(corrected_idx))
    print(corrected_labels, len(corrected_labels))
    print(clean_constraint_buffer, len(clean_constraint_buffer))
    print(noisy_constraint_buffer, len(noisy_constraint_buffer))
    np.savez(osp.join(dataloader.cache_root, f'constraint-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.npz'),
     clean_constraint_buffer=clean_constraint_buffer, noisy_constraint_buffer=noisy_constraint_buffer,
     idx=corrected_idx, labels=corrected_labels)
    return jsonify({
        'done': True,
    })

@api.route('/add_R', methods=['POST'])
def add_R():
    row_index = request.json['row_index']
    row_label = request.json['row_label']
    ret = dataloader.add_R(row_index, row_label)
    return jsonify(ret)


@api.route('/replace_R', methods=['POST'])
def replace_R():
    row_index = request.json['row_index']
    row_label = request.json['row_label']
    ret = dataloader.replace_R(row_index, row_label)
    return jsonify(ret)


@api.route('/set_gamma', methods=['POST'])
def set_gamma():
    gamma = request.json['gamma']
    dataloader.set_gamma(gamma)
    print(gamma, request.json, 'set gamma', dataloader.gamma)
    return jsonify({'status': 'success'})