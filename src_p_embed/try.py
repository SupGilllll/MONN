from collections import defaultdict
import numpy as np
import torch
import os
import pickle
os.chdir('/data/zhao/MONN/src_p_embed')
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

# uniprot_id_list = []
# max_len = 0
# with open('../create_dataset/out2_pdbbind_all_datafile.tsv', 'r') as f:
#     for line in f.readlines():
#         pdbid, pid, cid, inchi, seq, measure, label = line.strip().split('\t')
#         uniprot_id_list.append(pid)
# fasta_id_list = []
# with open('./target_uniprot_all.fasta', 'r') as f:
#     for line in f.readlines():
#         if line.startswith('>'):
#             fasta_id_list.append(line.strip()[1:])
# assert(len(fasta_id_list) == len(set(fasta_id_list)))
# if set(fasta_id_list) == set(uniprot_id_list):
#     print('identical')

# data = torch.load('./p_embed/B6HWK0.pt')
# embedding = data['representations'][33]
# print('happy')

max_len = 0
cnt = 0
pid = ''
seq = ''
pid_len_dict = {}
with open('./uniprot_all.fasta', 'r') as f:
    for line in f.readlines():
        if line[0] == '>':
            max_len = max(max_len, len(seq))
            if len(seq) > 1500:
                cnt += 1
            if pid != '':
                pid_len_dict[pid] = len(seq)
            pid = line.strip()[1:]
            seq = '' 
        else:
            seq += line.strip()
    pid_len_dict[pid] = len(seq)
with open('./pid_len_dict', 'wb') as f:
    pickle.dump(pid_len_dict, f)
