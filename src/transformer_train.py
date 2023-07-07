import argparse
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
import optuna

from transformer_utils import *
from transformer_model import *

# no RNN
#train and evaluate

def train_and_eval(train_data, valid_data, test_data, params):
    measure, setting, clu_thre, embedding, activation, n_rep, epochs, batch_size, lr, num_encoder_layers, \
    num_decoder_layers, d_encoder, d_decoder, d_model, dim_feedforward, nhead = params
    init_atoms, _, init_residues = loading_emb(measure, embedding)
    net = Transformer(init_atoms = init_atoms, init_residues = init_residues, 
                      d_encoder = d_encoder, d_decoder = d_decoder, d_model = d_model,
                      nhead = nhead, num_encoder_layers = num_encoder_layers, activation = activation,
                      num_decoder_layers = num_decoder_layers, dim_feedforward = dim_feedforward).cuda()  
    net.train()
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('total num params', pytorch_total_params)

    criterion1 = nn.MSELoss()
    criterion2 = Masked_BCELoss()

    # optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-4, weight_decay=0, amsgrad=True)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    shuffle_index = np.arange(len(train_data[0]))
    min_rmse = 1000
    for epoch in range(epochs):
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

            input_vertex, _, _, _, _, input_seq, _, affinity_label, pairwise_mask, pairwise_label = \
                [train_data[data_idx][shuffle_index[i * batch_size:(i+1)*batch_size]] for data_idx in range(10)]
            actual_batch_size = len(input_vertex)

            inputs = [input_vertex, input_seq]
            compound, protein, compound_mask, protein_mask = batch_data_process_transformer(inputs)

            affinity_label = torch.FloatTensor(affinity_label).cuda()
            pairwise_mask = torch.FloatTensor(pairwise_mask).cuda()
            pairwise_label = torch.FloatTensor(pad_label_2d(pairwise_label, compound, protein)).cuda()

            optimizer.zero_grad()
            affinity_pred, pairwise_pred = net(src = protein, tgt = compound, src_key_padding_mask = protein_mask, 
                                tgt_key_padding_mask = compound_mask, memory_key_padding_mask = protein_mask)

            loss_aff = criterion1(affinity_pred, affinity_label)
            loss_pairwise = criterion2(pairwise_pred, pairwise_label, pairwise_mask, compound_mask, protein_mask)
            loss = loss_aff + 0.1 * loss_pairwise

            total_loss += float(loss.data*actual_batch_size)
            affinity_loss += float(loss_aff.data*actual_batch_size)
            pairwise_loss += float(loss_pairwise.data*actual_batch_size)

            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()

        scheduler.step()
        loss_list = [total_loss, affinity_loss, pairwise_loss]
        loss_name = ['total loss', 'affinity loss', 'pairwise loss']
        print_loss = [loss_name[i] + ' ' + str(round(loss_list[i] / float(len(train_data[0])), 6)) for i in range(len(loss_name))]
        print('epoch:', epoch, ' '.join(print_loss))

        perf_name = ['RMSE', 'Pearson', 'Spearman', 'avg pairwise AUC']
        if epoch % 10 == 0:
            train_performance, train_label, train_output = test(net, train_data, batch_size)
            print_perf = [perf_name[i] + ' ' + str(round(train_performance[i], 6)) for i in range(len(perf_name))]
            print('train', len(train_output), ' '.join(print_perf))

        valid_performance, valid_label, valid_output = test(net, valid_data, batch_size)
        print_perf = [perf_name[i] + ' ' + str(round(valid_performance[i], 6)) for i in range(len(perf_name))]
        print('valid', len(valid_output), ' '.join(print_perf))

        if valid_performance[0] < min_rmse:
            min_rmse = valid_performance[0]
            test_performance, test_label, test_output = test(net, test_data, batch_size)
        print_perf = [perf_name[i] + ' ' + str(round(test_performance[i], 6)) for i in range(len(perf_name))]
        print('test ', len(test_output), ' '.join(print_perf))

    print('Finished Training')
    return test_performance, test_label, test_output


def test(net, test_data, batch_size):
    net.eval()
    output_list = []
    label_list = []
    pairwise_auc_list = []
    with torch.no_grad():
        for i in range(math.ceil(len(test_data[0])/batch_size)):
            input_vertex, _, _, _, _, input_seq, pids, aff_label, pairwise_mask, pairwise_label = \
                [test_data[data_idx][i*batch_size:(i+1)*batch_size] for data_idx in range(10)]
            
            inputs = [input_vertex, input_seq]
            compound, protein, compound_mask, protein_mask = batch_data_process_transformer(inputs)
            affinity_pred, pairwise_pred = net(src = protein, tgt = compound, src_key_padding_mask = protein_mask, 
                                               tgt_key_padding_mask = compound_mask, memory_key_padding_mask = protein_mask)
            
            vertex_mask = 1 - compound_mask.float()
            seq_mask = 1 - protein_mask.float()
            for j in range(len(pairwise_mask)):
                if pairwise_mask[j]:
                    num_vertex = int(torch.sum(vertex_mask[j, :]))
                    num_residue = int(torch.sum(seq_mask[j, :]))
                    pairwise_pred_i = pairwise_pred[j, :num_vertex, :num_residue].cpu().detach().numpy().reshape(-1)
                    pairwise_label_i = pairwise_label[j].reshape(-1)
                    pairwise_auc_list.append(roc_auc_score(pairwise_label_i, pairwise_pred_i))
                    # pairwise_pred_i = pairwise_pred[j, :num_vertex, surface_area_dict[pids[j]]].cpu().detach().numpy().reshape(-1)
                    # pairwise_label_i = pairwise_label[j][:, surface_area_dict[pids[j]]].reshape(-1)
                    # try:
                    #     pairwise_auc_list.append(roc_auc_score(pairwise_label_i, pairwise_pred_i))
                    # except ValueError:
                    #     pass
            output_list += affinity_pred.cpu().detach().numpy().reshape(-1).tolist()
            label_list += aff_label.reshape(-1).tolist()
    output_list = np.array(output_list)
    label_list = np.array(label_list)
    rmse_value, pearson_value, spearman_value = reg_scores(label_list, output_list)
    average_pairwise_auc = np.mean(pairwise_auc_list)

    test_performance = [rmse_value, pearson_value, spearman_value, average_pairwise_auc]
    return test_performance, label_list, output_list

def parse_args():
    parser = argparse.ArgumentParser(description = 'Pytorch Training Script')
    parser.add_argument('--cuda_device', type = int, default = 1)
    parser.add_argument('--measure', type = str, default = 'KIKD')
    parser.add_argument('--setting', type = str, default = 'new_protein')
    parser.add_argument('--clu_thre', type = float, default = 0.3)
    parser.add_argument('--embedding', type = str, default = 'blosum62')
    parser.add_argument('--activation', type = str, default = 'gelu')
    parser.add_argument('--optimizer', type = str, default = 'Adam')
    parser.add_argument('--n_rep', type = int, default = 5)
    parser.add_argument('--epochs', type = int, default = 40)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--lr', type = float, default = 5.5e-4)
    parser.add_argument('--num_layers', type = int, default = 2)
    # parser.add_argument('--d_encoder', type=int, default=20)
    parser.add_argument('--d_decoder', type = int, default = 82)
    parser.add_argument('--d_model', type = int, default = 256)
    parser.add_argument('--dim_feedforward', type = int, default = 512)
    parser.add_argument('--nhead', type = int, default = 4)

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
    n_rep = args.n_rep
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    num_encoder_layers = args.num_layers
    num_decoder_layers = args.num_layers
    d_encoder = {'blosum62': 20, 'one-hot': 20}
    d_decoder = args.d_decoder
    d_model = args.d_model
    # dim_feedforward = args.dim_feedforward
    dim_feedforward = 2 * d_model
    nhead = args.nhead
    if setting == 'new_compound' or setting == 'new_protein':
        n_fold = 5
    elif setting == 'new_new':
        n_fold = 9

    assert setting in ['new_compound', 'new_protein', 'new_new', 'imputation']
    assert clu_thre in [0.3, 0.4, 0.5, 0.6]
    assert measure in ['IC50', 'KIKD']
    assert embedding in ['blosum62', 'one-hot']
    
    para_names = ['measure', 'setting', 'clu_thre', 'embedding', 'activation', 'n_rep', 'epochs', 'batch_size', 
                  'lr', 'num_encoder_layers', 'num_decoder_layers', 'd_encoder', 'd_decoder', 'd_model', 
                  'dim_feedforward', 'nhead']
    params = [measure, setting, clu_thre, embedding, activation, n_rep, epochs, batch_size, lr, num_encoder_layers,
              num_decoder_layers, d_encoder[embedding], d_decoder, d_model, dim_feedforward, nhead]

    # print evaluation scheme
    print('Dataset: PDBbind v2021 with measurement', measure)
    print('Experiment setting', setting)
    print('Clustering threshold:', clu_thre)
    print('Protein embedding', embedding)
    print('Number of epochs:', epochs)
    print('Number of repeats:', n_rep)
    print('Hyper-parameters:', [para_names[i] + ' : ' + str(params[i]) for i in range(len(para_names))])
    with open('../preprocessing/surface_area_dict', 'rb') as f:
        surface_area_dict = pickle.load(f)
    all_start_time = time.time()

    rep_all_list = []
    rep_avg_list = []
    for a_rep in range(n_rep):
        rep_start_time = time.time()
        # load data
        data_pack, train_idx_list, valid_idx_list, test_idx_list = load_data(measure, setting, clu_thre, n_fold)
        fold_score_list = []

        for a_fold in range(n_fold):
            fold_start_time = time.time()
            print('repeat', a_rep+1, 'fold', a_fold+1, 'begin')
            train_idx, valid_idx, test_idx = train_idx_list[a_fold], valid_idx_list[a_fold], test_idx_list[a_fold]
            print('train num:', len(train_idx), 'valid num:', len(valid_idx), 'test num:', len(test_idx))

            train_data = data_from_index(data_pack, train_idx)
            valid_data = data_from_index(data_pack, valid_idx)
            test_data = data_from_index(data_pack, test_idx)

            test_performance, test_label, test_output = train_and_eval(train_data, valid_data, test_data, params)
            rep_all_list.append(test_performance)
            fold_score_list.append(test_performance)
            print('-'*30)
            print(f'repeat {a_rep + 1}, fold {a_fold + 1}, spend {format((time.time() - fold_start_time) / 3600.0, ".3f")} hours')
        print(f'repeat {a_rep + 1}, spend {format((time.time() - rep_start_time) / 3600.0, ".3f")}')
        print('fold avg performance', np.mean(fold_score_list, axis=0))
        rep_avg_list.append(np.mean(fold_score_list, axis=0))
        # np.save('MONN_rep_all_list_'+measure+'_'+setting+'_thre'+str(clu_thre), rep_all_list)

    print(f'whole training process spend {format((time.time() - all_start_time) / 3600.0, ".3f")}')
    print('all repetitions done')
    print('print all stats: RMSE, Pearson, Spearman, avg pairwise AUC')
    print('mean', np.mean(rep_all_list, axis=0))
    print('std', np.std(rep_all_list, axis=0))
    print('==============')
    print('print avg stats:  RMSE, Pearson, Spearman, avg pairwise AUC')
    print('mean', np.mean(rep_avg_list, axis=0))
    print('std', np.std(rep_avg_list, axis=0))
    # np.save('CPI_rep_all_list_'+measure+'_'+setting+'_thre'+str(clu_thre)+'_'+'_'.join(map(str,params)), rep_all_list)
    # np.save('MONN_rep_all_list_'+measure+'_'+setting+'_thre'+str(clu_thre), rep_all_list)
    return np.mean(rep_all_list, axis=0)[0]

def objective(trail):
    args.lr = trail.suggest_float('lr', 1e-4, 1e-3, step = 5e-5)
    args.num_layers = trail.suggest_int('num_layers', 1, 2)
    args.d_model = trail.suggest_int('hidden_dim', 128, 256, step = 16)
    args.nhead = trail.suggest_categorical('attention_heads', [1, 2, 4])
    args.activation = trail.suggest_categorical('activation_func', ['elu', 'leaky_relu', 'gelu'])
    # args.optimizer = trail.suggest_categorical('optimizer', ['Adam', 'RAdam', 'SGD'])
    rmse = main(args)
    return rmse

if __name__ == "__main__":
    os.chdir('/data/zhao/MONN/src')
    args = parse_args()
    # st = time.time()
    # study = optuna.create_study(study_name='Transformer Model Training', direction='minimize')
    # study.optimize(objective, n_trials = 120)
    # print(study.best_params)
    # print(study.best_trial)
    # print(study.best_trial.value)
    # print(format((time.time() - st) / 3600.0, ".3f"))
    # fig1 = optuna.visualization.plot_contour(study)
    # fig2 = optuna.visualization.plot_optimization_history(study)
    # fig3 = optuna.visualization.plot_param_importances(study)
    # fig1.write_html('../results/0710/contour.html')
    # fig2.write_html('../results/0710/optimization_history.html') 
    # fig3.write_html('../results/0710/param_importances.html') 
    main(args)