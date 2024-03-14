import sys
import math
import time
import pickle
import numpy as np
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from pdbbind_utils import *
from CPI_model import *

# no RNN
#train and evaluate

def train_and_eval(test_data, params, batch_size=32, num_epoch=30):
    init_A, init_B, init_W = loading_emb(measure, embedding)
    net = Net(init_A, init_B, init_W, params).cuda()
    net.load_state_dict(torch.load(f'../results/240116/transformer/save_model/baseline_{clu_thre}.pth'))
    net.eval()
    records = dict()
    pairwise_auc_list = []
    with torch.no_grad():
        for i in range(math.ceil(len(test_data[0])/batch_size)):
            input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, pids, affinity_label, pairwise_mask, pairwise_label, pdbids = \
                [test_data[data_idx][i*batch_size:(i+1)*batch_size] for data_idx in range(11)]
            
            inputs = [input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq]
            vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence = batch_data_process(inputs)
            affinity_pred, pairwise_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)
            
            for j in range(len(pairwise_mask)):
                if pairwise_mask[j]:
                    records[pdbids[j]] = dict()
                    num_vertex = int(torch.sum(vertex_mask[j, :]))
                    num_residue = int(torch.sum(seq_mask[j, :]))
                    pairwise_pred_i = pairwise_pred[j, :num_vertex, :num_residue].cpu().detach().numpy()
                    pairwise_label_i = pairwise_label[j]
                    records[pdbids[j]]['aff_pred'] = affinity_pred[j].cpu().detach().numpy().item()
                    records[pdbids[j]]['aff_label'] = affinity_label[j].item()
                    records[pdbids[j]]['error'] = abs(records[pdbids[j]]['aff_pred'] - records[pdbids[j]]['aff_label'])
                    records[pdbids[j]]['int_pred'] = pairwise_pred_i
                    records[pdbids[j]]['int_label'] = pairwise_label_i
                    score = roc_auc_score(pairwise_label_i.reshape(-1), pairwise_pred_i.reshape(-1))
                    pairwise_auc_list.append(score)
                    records[pdbids[j]]['auc'] = score
    pairwise_auc_score = np.mean(pairwise_auc_list)
    print(pairwise_auc_score)

    return records

if __name__ == "__main__":
    setup_seed()
    torch.cuda.set_device(1)
    # os.chdir('/data/zhao/MONN/src')
    measure = 'ALL'  # IC50 or KIKD
    setting = 'new_new'   # new_compound, new_protein or new_new
    clu_thre = 0.4  # 0.3, 0.4, 0.5 or 0.6
    embedding = 'blosum62'

    # measure = sys.argv[1]  # IC50 or KIKD
    # setting = sys.argv[2]   # new_compound, new_protein or new_new
    # clu_thre = float(sys.argv[3])  # 0.3, 0.4, 0.5 or 0.6
    # embedding = sys.argv[4]
    n_epoch = 30
    n_rep = 1

    assert setting in ['new_compound', 'new_protein', 'new_new', 'imputation']
    assert clu_thre in [0.3, 0.4, 0.5, 0.6]
    assert measure in ['IC50', 'KIKD', 'ALL']
    assert embedding in ['blosum62', 'one-hot']
    GNN_depth, inner_CNN_depth, DMA_depth = 4, 2, 2
    if setting == 'new_compound':
        n_fold = 5
        batch_size = 32
        k_head, kernel_size, hidden_size1, hidden_size2 = 2, 7, 128, 128
    elif setting == 'new_protein':
        n_fold = 5
        batch_size = 32
        k_head, kernel_size, hidden_size1, hidden_size2 = 1, 5, 128, 128
    elif setting == 'new_new':
        n_fold = 9
        batch_size = 32
        k_head, kernel_size, hidden_size1, hidden_size2 = 1, 7, 128, 128
    para_names = ['GNN_depth', 'inner_CNN_depth', 'DMA_depth',
                  'k_head', 'kernel_size', 'hidden_size1', 'hidden_size2']

    params = [GNN_depth, inner_CNN_depth, DMA_depth,
              k_head, kernel_size, hidden_size1, hidden_size2]
    #params = sys.argv[4].split(',')
    #params = map(int, params)

    # with open('../preprocessing/surface_area_dict', 'rb') as f:
    #     surface_area_dict = pickle.load(f)
    with open('../data/pocket_dict', 'rb') as f:
        pocket_area_dict = pickle.load(f)
    # print evaluation scheme
    print('Dataset: PDBbind v2021 with measurement', measure)
    print('Experiment setting', setting)
    print('Clustering threshold:', clu_thre)
    print('Protein embedding', embedding)
    print('Number of epochs:', n_epoch)
    print('Number of repeats:', n_rep)
    print('Hyper-parameters:', [para_names[i] + ':' + str(params[i]) for i in range(7)])
    all_start_time = time.time()

    rep_all_list = []
    rep_avg_list = []
    for a_rep in range(n_rep):
        rep_start_time = time.time()
        # load data
        data_pack, train_idx_list, valid_idx_list, test_idx_list = load_data(
            measure, setting, clu_thre, n_fold)
        fold_score_list = []
        accumulated_interaction_label = []
        accumulated_interaction_pred = []

        for a_fold in [1]:
            setup_seed()
            fold_start_time = time.time()
            print('repeat', a_rep+1, 'fold', a_fold+1, 'begin')
            train_idx, valid_idx, test_idx = train_idx_list[a_fold], valid_idx_list[a_fold], test_idx_list[a_fold]
            print('train num:', len(train_idx), 'valid num:', len(valid_idx), 'test num:', len(test_idx))

            test_data = data_from_index(data_pack, test_idx)
            records = train_and_eval(test_data, params, batch_size, n_epoch)
            np.save('../results/240116/transformer/load_model/baseline_records', records)
            print('-'*30)
            print(f'repeat {a_rep + 1}, fold {a_fold + 1}, spend {format((time.time() - fold_start_time) / 3600.0, ".3f")} hours')
