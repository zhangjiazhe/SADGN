import torch
import numpy as np
import pandas as pd
import datetime
import  os


def generate_graph_seq2seq_data(data, data_adj, x_offsets, y_offsets):
    num_samples = data.shape[0]
    x, y, x_adj, y_adj = [], [], [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        x_adj.append(data_adj[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
        y_adj.append(data_adj[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    x_adj = np.stack(x_adj, axis=0)
    y_adj = np.stack(y_adj, axis=0)
    return x, y, x_adj, y_adj

def generate_train_val_test(feature_path, adj_path, data_dir, seq_length_x, seq_length_y):

    data, data_adj = np.load(feature_path), np.load(adj_path)

    print(data.shape, data_adj.shape)

    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(1, (seq_length_y + 1), 1))
    x, y, x_adj, y_adj = generate_graph_seq2seq_data(data, data_adj, x_offsets, y_offsets)

    print(x.shape, y.shape, x_adj.shape, y_adj.shape)

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    x_train, y_train, xadj_train, yadj_train = x[:num_train], y[:num_train], x_adj[:num_train], y_adj[:num_train]
    x_val, y_val, xadj_val, yadj_val = x[num_train : num_train + num_val], y[num_train : num_train + num_val],x_adj[num_train : num_train + num_val], y_adj[num_train : num_train + num_val]
    x_test, y_test, xadj_test, yadj_test = x[-num_test:], y[-num_test:], x_adj[-num_test:], y_adj[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y, _xadj, _yadj = locals()["x_" + cat], locals()["y_" + cat],locals()["xadj_" + cat], locals()["yadj_" + cat]
        print(cat,"x: ", _x.shape, "y: ", _y.shape, "x_adj: ", _xadj.shape, "y_adj: ", _yadj.shape)
        np.savez_compressed(
            os.path.join(data_dir, f"{cat}.npz"),
            x = _x,
            y = _y,
            x_adj = _xadj,
            y_adj = _yadj,
            x_offsets = x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets = y_offsets.reshape(list(y_offsets.shape) + [1]),

        )

if __name__ == '__main__':

   feature_path = './data/rawdata/feature.npy'
   adj_path = './data/rawdata/s_t_adj.npy'
   data_dir_path = './data/TH/'

   generate_train_val_test(feature_path=feature_path, adj_path=adj_path, data_dir=data_dir_path, seq_length_x=14, seq_length_y=7)






