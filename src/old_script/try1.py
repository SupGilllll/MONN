from collections import defaultdict
import numpy as np
# import torch
import os
import pickle
import json
import statistics
import matplotlib.pyplot as plt
# os.chdir('/data/zhao/MONN/src')
# aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
# word_dict = defaultdict(lambda: len(word_dict))
# for aa in aa_list:
#         word_dict[aa]
# word_dict['X']
# def Protein2Sequence(sequence, ngram=1):
#     # convert sequence to CNN input
#     sequence = sequence.upper()
#     word_list = [sequence[i:i+ngram] for i in range(len(sequence)-ngram+1)]
#     output = []
#     for word in word_list:
#         if word not in aa_list:
#             output.append(word_dict['X'])
#         else:
#             output.append(word_dict[word])
#     if ngram == 3:
#         output = [-1]+output+[-1] # pad
#     return np.array(output, np.int32)
# seq = 'ACDEFGCDEFG'
# data = Protein2Sequence(seq)
# print('happy')

# data = torch.load('./test_emb_esm2/1.pt')
# print('happy')

# with open('./pid_len_dict', 'rb') as f:
#     data = pickle.load(f)
# cnt = 0
# for pid, len in data.items():
#     if len > 5000:
#         cnt += 1
#         print(pid)
# print(cnt)

# with open('../preprocessing/RSA.json', 'r') as f:
#     data = json.load(f)
# print(list(data[0].keys()))

# with open('../create_dataset/out8_final_pocket_dict', 'rb') as f:
#     pocket_dict = pickle.load(f)
# with open('../preprocessing/surface_area_dict', 'rb') as f:
#     surface_area_dict = pickle.load(f)
# with open('../data/interaction_dict','rb') as f:
#     interaction_dict = pickle.load(f)

# interact_cnt = 0
# interact_total = 0
# interact_percent = []
# sample1 = 0
# sample2 = 0
# pocket_cnt = 0
# pocket_total = 0
# pocket_percent = []

# for sample in interaction_dict.values():
#     pid = sample['uniprot_id']
#     if pid not in surface_area_dict:
#         continue
#     sample1 += 1
#     interact_total += len(sample['interact_in_uniprot_seq'])
#     temp_cnt = 0
#     for residue in sample['interact_in_uniprot_seq']:
#         if residue in surface_area_dict[pid]:
#             interact_cnt += 1
#             temp_cnt += 1
#     interact_percent.append(temp_cnt / len(sample['interact_in_uniprot_seq']))

# for sample in pocket_dict.values():
#     pid = sample['uniprot_id']
#     if pid not in surface_area_dict:
#         continue
#     sample2 += 1
#     pocket_total += len(sample['pocket_in_uniprot_seq'])
#     temp_cnt = 0
#     for residue in sample['pocket_in_uniprot_seq']:
#         if residue in surface_area_dict[pid]:
#             pocket_cnt += 1
#             temp_cnt += 1
#     pocket_percent.append(temp_cnt / len(sample['pocket_in_uniprot_seq']))

# print(sample1, sample2)
# print(interact_total, interact_cnt)
# print(pocket_total, pocket_cnt)
# print(statistics.mean(interact_percent))
# print(statistics.mean(pocket_percent))

affinity_list = []
pairwise_list = []
rmse_list = []
auc_list = []
with open('../results/0626/blosum62/KIKD_new_protein_0.3.log', 'r') as f:
    for line in f.readlines()[:1430]:
        if "affinity loss" in line:
            aff_loss, pw_loss = line.strip().split()[7], line.strip().split()[10]
            affinity_list.append(float(aff_loss))
            pairwise_list.append(float(pw_loss))
            continue
        if line.startswith('test'):
            rmse, auc = line.strip().split()[3], line.strip().split()[11]
            rmse_list.append(float(rmse))
            auc_list.append(float(auc))

plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.plot(auc_list)
plt.savefig('plot.png')