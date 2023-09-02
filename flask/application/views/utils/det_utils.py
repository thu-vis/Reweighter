import numpy as np

def intersect(box_a, box_b):
    A = box_a.shape[0]
    B = box_b.shape[0]
    min_xy_a = box_a[:, :2][:, np.newaxis, :].repeat(axis=1, repeats=B)
    min_xy_b = box_b[:, :2][np.newaxis, :, :].repeat(axis=0, repeats=A)
    max_xy_a = box_a[:, 2:4][:, np.newaxis, :].repeat(axis=1, repeats=B)
    max_xy_b = box_b[:, 2:4][np.newaxis, :, :].repeat(axis=0, repeats=A)
    max_xy = np.minimum(max_xy_a, max_xy_b)
    min_xy = np.maximum(min_xy_a, min_xy_b)
    inter = (max_xy - min_xy).clip(0)
    areas = inter[:, :, 0] * inter[:, :, 1]
    return areas

def jacard(box_pred, box_truth):
    A = box_pred.shape[0]
    B = box_truth.shape[0]
    if B == 0:
        return np.zeros((A, 1))
    inter = intersect(box_pred, box_truth)
    area_a = ((box_pred[:, 2]-box_pred[:, 0]) *
              (box_pred[:, 3]-box_pred[:, 1]))[:, np.newaxis].repeat(axis=1, repeats=B)
    area_b = ((box_truth[:, 2]-box_truth[:, 0]) *
              (box_truth[:, 3]-box_truth[:, 1]))[np.newaxis, :].repeat(axis=0, repeats=A)
    union = area_a + area_b - inter
    return inter / union  # [A,B]