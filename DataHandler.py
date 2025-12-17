import os
import time

import numpy as np
import scipy
import scipy.sparse as sp
import torch
from Args import args
from DataLoader import *
from Log import log_print
from torch.utils.data import DataLoader


import os
import numpy as np
import scipy.sparse

class DataHandler: 
    def __init__(self): 
        self.data_directory = os.path.join(args.data_dir, args.data)

        trn_path = self.get_data_path("train")
        tst_path = self.get_data_path("test")

        trn_data = self.load_interactions(trn_path)
        tst_data = self.load_interactions(tst_path)

        users_trn, items_trn = trn_data[:, 0], trn_data[:, 1]
        users_tst, items_tst = tst_data[:, 0], tst_data[:, 1]

        self.n_user = int(max(np.max(users_trn), np.max(users_tst))) + 1
        self.n_item = int(max(np.max(items_trn), np.max(items_tst))) + 1

        adj_trn = self.build_adj(trn_data)
        adj_tst = self.build_adj(tst_data)

        self.graph = self.getSparseGraph(adj_trn)
        self.loadData(adj_trn, adj_tst)

    def get_data_path(self, prefix):
        npy_name = f"{prefix}_list.npy"
        txt_name = f"{prefix}.txt"
        npy_path = os.path.join(self.data_directory, npy_name)
        txt_path = os.path.join(self.data_directory, txt_name)

        if os.path.isfile(npy_path):
            return npy_path
        elif os.path.isfile(txt_path):
            return txt_path
        else:
            raise FileNotFoundError(f"Neither {npy_name} nor {txt_name} found in {self.data_directory}")

    def load_interactions(self, path):
        if path.endswith('.npy'):
            return np.load(path)
        elif path.endswith('.txt'):
            return np.loadtxt(path, dtype=int)
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def build_adj(self, interactions):
        users = interactions[:, 0]
        items = interactions[:, 1]
        data = np.ones(len(users))
        adj = scipy.sparse.csr_matrix((data, (users, items)), shape=(self.n_user, self.n_item))
        return adj

    def getSparseGraph(self, adj_trn):
        log_print("loading adjacency matrix")
        try:
            pre_adj_mat = sp.load_npz(self.data_directory + "/s_pre_adj_mat.npz")
            log_print("successfully loaded...")
            norm_adj = pre_adj_mat
        except:
            log_print("generating adjacency matrix")
            s = time.time()
            adj_mat = sp.dok_matrix(
                (self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32
            )
            adj_mat = adj_mat.tolil()
            R = adj_trn.tocsr().tolil()
            adj_mat[: self.n_user, self.n_user :] = R
            adj_mat[self.n_user :, : self.n_user] = R.T
            adj_mat = adj_mat.todok()

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time.time()
            log_print(f"costing {end - s}s, saved norm_mat...")
            sp.save_npz(self.data_directory + "/s_pre_adj_mat.npz", norm_adj)
        Graph = self.convert_sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce()
        log_print("don't split the matrix")
        return Graph

    def loadData(self, adj_trn, adj_tst):
        train_dataset = TrnData(adj_trn)
        tst_dataset = TstData(adj_tst, adj_trn)
        self.trnLoader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6
        )
        self.tstLoader = DataLoader(
            tst_dataset, batch_size=args.tstBat, shuffle=False, num_workers=6
        )

    def convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
