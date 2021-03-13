import numpy as np
import torch



def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def unmask(X, M):
    if X[0].ndim == 1 or (X[0].shape[0] > X[0].shape[1]):
        return [X[i][M[i].flatten() > 0] for i in range(len(X))]
    else:
        return [X[i][:, M[i].flatten() > 0] for i in range(len(X))]

def segment_labels(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
    return Yi_split


def segment_data(Xi, Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    Xi_split = [np.squeeze(Xi[:, idxs[i]:idxs[i + 1]]) for i in range(len(idxs) - 1)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
    return Xi_split, Yi_split


def segment_intervals(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    intervals = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
    return intervals


def segment_lengths(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    intervals = [(idxs[i + 1] - idxs[i]) for i in range(len(idxs) - 1)]
    return np.array(intervals)

# TODO:加快代码运行，暂不清楚机制
# @jit("int64[:](int64[:], int64)")
def partition_latent_labels(Yi, n_latent):
    if n_latent == 1:
        return Yi

    Zi = Yi.copy()
    intervals = segment_intervals(Yi)
    for i in range(len(intervals)):
        y = Zi[intervals[i][0]]
        diff = intervals[i][1] - intervals[i][0]
        partition_size = diff // n_latent
        start = intervals[i][0]
        stop = start + partition_size
        for j in range(0, n_latent - 1):
            Zi[start:stop] = Yi[start] * n_latent + j
            start += partition_size
            stop += partition_size

        # Do last partition differently in case of rounding errors
        stop = intervals[i][1]
        Zi[start:stop] = (Yi[start] + 1) * n_latent - 1

    return Zi