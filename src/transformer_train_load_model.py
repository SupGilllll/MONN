import argparse
import math
import time
import pickle
import numpy as np
import os

import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import optuna

from transformer_utils import *
from transformer_model_graph import *

# no RNN
#train and evaluate

def train_and_eval(test_data, params):
    measure, setting, clu_thre, embedding, activation, opt, sch, n_rep, epochs, batch_size, lr, num_encoder_layers, \
    num_decoder_layers, d_encoder, d_decoder, d_model, dim_feedforward, nhead = params
    init_atoms, init_bonds, init_residues = loading_emb(measure, 'blosum62')
    net = Transformer(init_atoms = init_atoms, init_bonds = init_bonds, init_residues = init_residues, 
                      d_encoder = d_encoder, d_decoder = d_decoder, d_model = d_model,
                      nhead = nhead, num_encoder_layers = num_encoder_layers, activation = activation,
                      num_decoder_layers = num_decoder_layers, dim_feedforward = dim_feedforward).cuda() 
    net.load_state_dict(torch.load(f'/data/zhao/MONN/results/240116/transformer/save_model/{clu_thre}.pth'))
    net.eval()
    records = []
    pairwise_auc_list = []
    with torch.no_grad():
        for i in range(math.ceil(len(test_data[0])/batch_size)):
            input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, \
            input_seq, pids, affinity_label, pairwise_mask, pairwise_label, pdbids = \
                [test_data[data_idx][i*batch_size:(i+1)*batch_size] for data_idx in range(11)]
            actual_batch_size = len(input_vertex)
            
            inputs = [input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq]
            compound, edge, atom_adj, bond_adj, protein, nbs_mask, compound_mask, protein_mask = batch_data_process_transformer_graph(inputs)

            affinity_pred, pairwise_pred = net(src = protein, tgt = compound, edge = edge, atom_adj = atom_adj, bond_adj = bond_adj,
                                               nbs_mask = nbs_mask, pids = pids, src_key_padding_mask = protein_mask, 
                                               tgt_key_padding_mask = compound_mask, memory_key_padding_mask = protein_mask)
            
            vertex_mask = 1 - compound_mask.float()
            seq_mask = 1 - protein_mask.float()
            for j in range(len(pairwise_mask)):
                record = dict()
                if pairwise_mask[j]:
                    num_vertex = int(torch.sum(vertex_mask[j, :]))
                    num_residue = int(torch.sum(seq_mask[j, :]))
                    pairwise_pred_i = pairwise_pred[j, :num_vertex, :num_residue].cpu().detach().numpy()
                    pairwise_label_i = pairwise_label[j]
                    record['pdbid'] = pdbids[j]
                    record['aff_pred'] = affinity_pred[j].cpu().detach().numpy().item()
                    record['aff_label'] = affinity_label[j].item()
                    record['error'] = abs(record['aff_pred'] - record['aff_label'])
                    record['int_pred'] = pairwise_pred_i
                    record['int_label'] = pairwise_label_i
                    score = roc_auc_score(pairwise_label_i.reshape(-1), pairwise_pred_i.reshape(-1))
                    pairwise_auc_list.append(score)
                    record['auc'] = score
                    records.append(record)
                    # pocket_area_list = pocket_area_dict[pdbids[j]]['pocket_in_uniprot_seq']
                    # pairwise_pred_i = pairwise_pred[j, :num_vertex, pocket_area_list].cpu().detach().numpy().reshape(-1)
                    # pairwise_label_i = pairwise_label[j][:, pocket_area_list].reshape(-1)
                    # try:
                    #     pairwise_auc_list.append(roc_auc_score(pairwise_label_i, pairwise_pred_i))
                    # except ValueError:
                    #     pass
    pairwise_auc_score = np.mean(pairwise_auc_list)
    print(pairwise_auc_score)
    return records


def parse_args():
    parser = argparse.ArgumentParser(description = 'Pytorch Training Script')
    parser.add_argument('--cuda_device', type = int, default = 1)
    parser.add_argument('--measure', type = str, default = 'ALL')
    parser.add_argument('--setting', type = str, default = 'new_new')
    parser.add_argument('--clu_thre', type = float, default = 0.4)
    parser.add_argument('--embedding', type = str, default = 'blosum62')
    parser.add_argument('--activation', type = str, default = 'elu')
    parser.add_argument('--optimizer', type = str, default = 'RAdam')
    parser.add_argument('--scheduler', type = str, default = 'StepLR_10')
    parser.add_argument('--n_rep', type = int, default = 1)
    parser.add_argument('--epochs', type = int, default = 30)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--lr', type = float, default = 8e-4)
    parser.add_argument('--pos_encoding', type = str, default = 'none')
    parser.add_argument('--encoder_layers', type = int, default = 2)
    parser.add_argument('--decoder_layers', type = int, default = 4)
    # parser.add_argument('--d_encoder', type=int, default=20)
    parser.add_argument('--d_decoder', type = int, default = 82)
    parser.add_argument('--d_model', type = int, default = 224)
    parser.add_argument('--dim_feedforward', type = int, default = 448)
    parser.add_argument('--nhead', type = int, default = 4)
    parser.add_argument('--loss_weight', type = float, default = 0.7)

    args = parser.parse_args()
    return args

def main(args):     
    setup_seed()
    torch.cuda.set_device(args.cuda_device)
    measure = args.measure  # IC50 or KIKD
    setting = args.setting   # new_compound, new_protein or new_new
    clu_thre = args.clu_thre  # 0.3, 0.4, 0.5 or 0.6
    embedding = args.embedding
    activation = args.activation
    optimizer = args.optimizer
    scheduler = args.scheduler
    n_rep = args.n_rep
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    num_encoder_layers = args.encoder_layers
    num_decoder_layers = args.decoder_layers
    d_encoder = {'blosum62' : 20, 'one-hot' : 20, 't33' : 1280}
    d_decoder = args.d_decoder
    d_model = args.d_model
    # dim_feedforward = args.dim_feedforward
    dim_feedforward = 2 * d_model
    nhead = args.nhead
    global loss_weight 
    loss_weight = args.loss_weight
    if setting == 'new_compound' or setting == 'new_protein':
        n_fold = 5
    elif setting == 'new_new':
        n_fold = 9

    assert setting in ['new_compound', 'new_protein', 'new_new', 'imputation']
    assert clu_thre in [0.3, 0.4, 0.5, 0.6]
    assert measure in ['IC50', 'KIKD', 'ALL']
    assert embedding in ['blosum62', 'one-hot', 't33']
    
    para_names = ['measure', 'setting', 'clu_thre', 'embedding', 'activation', 'optimizer', 'scheduler', 
                  'n_rep', 'epochs', 'batch_size', 'lr', 'num_encoder_layers', 'num_decoder_layers', 
                  'd_encoder', 'd_decoder', 'd_model', 'dim_feedforward', 'nhead']
    params = [measure, setting, clu_thre, embedding, activation, optimizer, scheduler, n_rep, 
              epochs, batch_size, lr, num_encoder_layers, num_decoder_layers, d_encoder[embedding], 
              d_decoder, d_model, dim_feedforward, nhead]

    # print evaluation scheme
    print('Dataset: PDBbind v2021 with measurement', measure)
    print('Experiment setting', setting)
    print('Clustering threshold:', clu_thre)
    print('Protein embedding:', embedding)
    print('Number of epochs:', epochs)
    print('Number of repeats:', n_rep)
    print('Hyper-parameters:', [para_names[i] + ' : ' + str(params[i]) for i in range(len(para_names))])
    all_start_time = time.time()

    rep_all_list = []
    rep_avg_list = []
    for a_rep in range(n_rep):
        rep_start_time = time.time()
        # load data
        data_pack, train_idx_list, valid_idx_list, test_idx_list = load_data(measure, setting, clu_thre, n_fold)
        fold_score_list = []

        for a_fold in [1]:
            setup_seed()
            fold_start_time = time.time()
            print('repeat', a_rep+1, 'fold', a_fold+1, 'begin')
            train_idx, valid_idx, test_idx = train_idx_list[a_fold], valid_idx_list[a_fold], test_idx_list[a_fold]
            print('train num:', len(train_idx), 'valid num:', len(valid_idx), 'test num:', len(test_idx))

            test_data = data_from_index(data_pack, test_idx)
            records = train_and_eval(test_data, params)
            np.save('/data/zhao/MONN/results/240116/transformer/load_model/records', records)
            print('-'*30)
            print(f'repeat {a_rep + 1}, fold {a_fold + 1}, spend {format((time.time() - fold_start_time) / 3600.0, ".3f")} hours')
        print(f'repeat {a_rep + 1}, spend {format((time.time() - rep_start_time) / 3600.0, ".3f")}')
        print('fold avg performance', np.mean(fold_score_list, axis=0))
        rep_avg_list.append(np.mean(fold_score_list, axis=0))
        # np.save('MONN_rep_all_list_'+measure+'_'+setting+'_thre'+str(clu_thre), rep_all_list)
        print("========== Whole AUC ==========")
        # print("Score: ", roc_auc_score(total_interaction_label, total_interaction_pred))
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
    # np.save('/data/zhao/MONN/results/240116/transformer/'+measure+'_'+setting+'_thre'+str(clu_thre)+'_aff_label', total_aff_label)
    # np.save('/data/zhao/MONN/results/240116/transformer/'+measure+'_'+setting+'_thre'+str(clu_thre)+'_aff_pred', total_aff_pred)
    # np.save('/data/zhao/MONN/results/240116/transformer/'+measure+'_'+setting+'_thre'+str(clu_thre)+'_label', total_interaction_label)
    # np.save('/data/zhao/MONN/results/240116/transformer/'+measure+'_'+setting+'_thre'+str(clu_thre)+'_pred', total_interaction_pred)
    # np.save('/data/zhao/MONN/results/240116/transformer/'+measure+'_'+setting+'_thre'+str(clu_thre)+'_auc_list', total_auc_score)
    # np.save('CPI_rep_all_list_'+measure+'_'+setting+'_thre'+str(clu_thre)+'_'+'_'.join(map(str,params)), rep_all_list)
    # np.save('MONN_rep_all_list_'+measure+'_'+setting+'_thre'+str(clu_thre), rep_all_list)
    return np.mean(rep_all_list, axis=0)[0]

if __name__ == "__main__":
    os.chdir('/data/zhao/MONN/src')
    args = parse_args()
    with open('/data/zhao/MONN/data/pocket_dict', 'rb') as f:
        pocket_area_dict = pickle.load(f)

    main(args)