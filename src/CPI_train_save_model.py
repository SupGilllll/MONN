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

def train_and_eval(train_data, valid_data, test_data, params, batch_size=32, num_epoch=30):
    init_A, init_B, init_W = loading_emb(measure, embedding)
    net = Net(init_A, init_B, init_W, params)
    net.cuda()
    net.apply(weights_init)
    net.train()
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('total num params', pytorch_total_params)

    criterion1 = nn.MSELoss()
    criterion2 = Masked_BCELoss()

    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=0.0005, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    shuffle_index = np.arange(len(train_data[0]))
    min_rmse = 1000
    #max_auc = 0
    for epoch in range(num_epoch):
        net.train()
        np.random.shuffle(shuffle_index)
        for param_group in optimizer.param_groups:
            print('learning rate:', param_group['lr'])

        total_loss = 0
        affinity_loss = 0
        pairwise_loss = 0

        for i in range(math.ceil(len(train_data[0])/batch_size)):
            if i % 20 == 0:
                print('epoch', epoch, 'batch', i)

            input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, _, affinity_label, pairwise_mask, pairwise_label = \
                [train_data[data_idx][shuffle_index[i * batch_size:(i+1)*batch_size]] for data_idx in range(10)]
            actual_batch_size = len(input_vertex)

            inputs = [input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq]
            vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence = batch_data_process(inputs)

            affinity_label = torch.FloatTensor(affinity_label).cuda()
            pairwise_mask = torch.FloatTensor(pairwise_mask).cuda()
            pairwise_label = torch.FloatTensor(pad_label_2d(pairwise_label, vertex, sequence)).cuda()

            optimizer.zero_grad()
            affinity_pred, pairwise_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)

            loss_aff = criterion1(affinity_pred, affinity_label)
            loss_pairwise = criterion2(pairwise_pred, pairwise_label, pairwise_mask, vertex_mask, seq_mask)
            loss = loss_aff + 0.1*loss_pairwise
            # print("training stage non-zero count", torch.count_nonzero(pairwise_pred >= 0.5), torch.count_nonzero(pairwise_label))
            # loss_aff = criterion1(affinity_pred, affinity_label)
            # loss_pairwise = criterion2(pairwise_pred, pairwise_label, pairwise_mask, vertex_mask, seq_mask)
            # loss = loss_aff

            total_loss += float(loss.data*actual_batch_size)
            affinity_loss += float(loss_aff.data*actual_batch_size)
            pairwise_loss += float(loss_pairwise.data*actual_batch_size)

            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()

        scheduler.step()
        loss_list = [total_loss, affinity_loss, pairwise_loss]
        loss_name = ['total loss', 'affinity loss', 'pairwise loss']
        print_loss = [loss_name[i]+' '+str(round(loss_list[i]/float(len(train_data[0])), 6)) for i in range(len(loss_name))]
        print('epoch:', epoch, ' '.join(print_loss))

        perf_name = ['RMSE', 'Pearson', 'Spearman', 'avg pairwise AUC', 'fold AUC']
        if epoch % 10 == 0:
            train_performance, _, _, _, _ = test(net, train_data, batch_size)
            print_perf = [perf_name[i]+' '+str(round(train_performance[i], 6)) for i in range(len(perf_name))]
            print('train', len(train_data[0]), ' '.join(print_perf))

        valid_performance, _, _, _, _ = test(net, valid_data, batch_size)
        print_perf = [perf_name[i]+' '+str(round(valid_performance[i], 6)) for i in range(len(perf_name))]
        print('valid', len(valid_data[0]), ' '.join(print_perf))

        if valid_performance[0] < min_rmse:
            torch.save(net.state_dict(), f'../results/240116/transformer/save_model/baseline_{clu_thre}.pth')
            # if valid_performance[-1] > max_auc:
            min_rmse = valid_performance[0]
            #max_auc = valid_performance[-1]
            test_performance, aff_label, aff_pred, interaction_label, interaction_pred = test(net, test_data, batch_size)
        print_perf = [perf_name[i]+' '+str(round(test_performance[i], 6)) for i in range(len(perf_name))]
        print('test ', len(test_data[0]), ' '.join(print_perf))

    print('Finished Training')
    return test_performance, aff_label, aff_pred, interaction_label, interaction_pred


def test(net, test_data, batch_size):
    net.eval()
    pairwise_auc_list = []
    accumulated_aff_pred = []
    accumulated_aff_label = []
    accumulated_interaction_pred = []
    accumulated_interaction_label = []
    with torch.no_grad():
        for i in range(math.ceil(len(test_data[0])/batch_size)):
            input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, pids, affinity_label, pairwise_mask, pairwise_label, pdbids = \
                [test_data[data_idx][i*batch_size:(i+1)*batch_size] for data_idx in range(11)]
            
            inputs = [input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq]
            vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence = batch_data_process(inputs)
            affinity_pred, pairwise_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)
            # print("test stage non-zero count", torch.count_nonzero(pairwise_pred >= 0.5))
            
            for j in range(len(pairwise_mask)):
                if pairwise_mask[j]:
                    num_vertex = int(torch.sum(vertex_mask[j, :]))
                    num_residue = int(torch.sum(seq_mask[j, :]))
                    pairwise_pred_i = pairwise_pred[j, :num_vertex, :num_residue].cpu().detach().numpy().reshape(-1)
                    pairwise_label_i = pairwise_label[j].reshape(-1)
                    pairwise_auc_list.append(roc_auc_score(pairwise_label_i, pairwise_pred_i))
                    accumulated_interaction_pred.append(pairwise_pred_i)
                    accumulated_interaction_label.append(pairwise_label_i)
                    # pocket_area_list = pocket_area_dict[pdbids[j]]['pocket_in_uniprot_seq']
                    # pairwise_pred_i = pairwise_pred[j, :num_vertex, pocket_area_list].cpu().detach().numpy().reshape(-1)
                    # pairwise_label_i = pairwise_label[j][:, pocket_area_list].reshape(-1)
                    # try:
                    #     pairwise_auc_list.append(roc_auc_score(pairwise_label_i, pairwise_pred_i))
                    # except ValueError:
                    #     pass
                    # pairwise_pred_i = pairwise_pred[j, :num_vertex, surface_area_dict[pids[j]]].cpu().detach().numpy().reshape(-1)
                    # pairwise_label_i = pairwise_label[j][:, surface_area_dict[pids[j]]].reshape(-1)
                    # try:
                    #     pairwise_auc_list.append(roc_auc_score(pairwise_label_i, pairwise_pred_i))
                    # except ValueError:
                    #     pass
            accumulated_aff_pred.append(affinity_pred.cpu().detach().numpy().reshape(-1))
            accumulated_aff_label.append(affinity_label.reshape(-1))
    aff_pred = np.concatenate(accumulated_aff_pred)
    aff_label = np.concatenate(accumulated_aff_label)
    interaction_pred = np.concatenate(accumulated_interaction_pred)
    interaction_label = np.concatenate(accumulated_interaction_label)
    rmse_value, pearson_value, spearman_value = reg_scores(aff_label, aff_pred)
    pairwise_auc_score = np.mean(pairwise_auc_list)
    # fold_auc_score = roc_auc_score(interaction_label, interaction_pred)

    test_performance = [rmse_value, pearson_value, spearman_value, pairwise_auc_score, 0]
    return test_performance, aff_label, aff_pred, interaction_label, interaction_pred


if __name__ == "__main__":
    setup_seed()
    torch.cuda.set_device(1)
    # os.chdir('/data/zhao/MONN/src')
    # measure = 'KIKD'  # IC50 or KIKD
    # setting = 'new_new'   # new_compound, new_protein or new_new
    # clu_thre = 0.4  # 0.3, 0.4, 0.5 or 0.6
    # embedding = 'blosum62'

    measure = sys.argv[1]  # IC50 or KIKD
    setting = sys.argv[2]   # new_compound, new_protein or new_new
    clu_thre = float(sys.argv[3])  # 0.3, 0.4, 0.5 or 0.6
    embedding = sys.argv[4]
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

            train_data = data_from_index(data_pack, train_idx)
            valid_data = data_from_index(data_pack, valid_idx)
            test_data = data_from_index(data_pack, test_idx)

            test_performance, aff_label, aff_pred, interaction_label, interaction_pred = train_and_eval(train_data, valid_data, test_data, params, batch_size, n_epoch)
            accumulated_interaction_label.append(interaction_label)
            accumulated_interaction_pred.append(interaction_pred)
            rep_all_list.append(test_performance)
            fold_score_list.append(test_performance)
            print('-'*30)
            print(f'repeat {a_rep + 1}, fold {a_fold + 1}, spend {format((time.time() - fold_start_time) / 3600.0, ".3f")} hours')
        total_interaction_label = np.concatenate(accumulated_interaction_label)
        total_interaction_pred = np.concatenate(accumulated_interaction_pred)
        print(f'repeat {a_rep + 1}, spend {format((time.time() - rep_start_time) / 3600.0, ".3f")}')
        print('fold avg performance', np.mean(fold_score_list, axis=0))
        rep_avg_list.append(np.mean(fold_score_list, axis=0))
        # np.save('MONN_rep_all_list_'+measure+'_'+setting+'_thre'+str(clu_thre), rep_all_list)
        print("========== Whole AUC ==========")
        print("Score: ", roc_auc_score(total_interaction_label, total_interaction_pred))
        print('==============')

    print(f'whole training process spend {format((time.time() - all_start_time) / 3600.0, ".3f")}')
    print('all repetitions done')
    print('print all stats: RMSE, Pearson, Spearman, avg pairwise AUC, fold AUC')
    print('mean', np.mean(rep_all_list, axis=0))
    print('std', np.std(rep_all_list, axis=0))
    print('==============')
    print('print avg stats:  RMSE, Pearson, Spearman, avg pairwise AUC, fold AUC')
    print('mean', np.mean(rep_avg_list, axis=0))
    print('std', np.std(rep_avg_list, axis=0))
    print('Hyper-parameters:', [para_names[i] + ':'+str(params[i]) for i in range(7)])
    # np.save('../results/240116/baseline/'+measure+'_'+setting+'_thre'+str(clu_thre)+'_label', total_interaction_label)
    # np.save('../results/240116/baseline/'+measure+'_'+setting+'_thre'+str(clu_thre)+'_pred', total_interaction_pred)
