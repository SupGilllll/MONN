from collections import defaultdict
import numpy as np
import torch
import os
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

uniprot_id_list = []
with open('../create_dataset/out2_pdbbind_all_datafile.tsv', 'r') as f:
    for line in f.readlines():
        pdbid, pid, cid, inchi, seq, measure, label = line.strip().split('\t')
        uniprot_id_list.append(pid)
fasta_id_list = []
with open('./target_uniprot_all.fasta', 'r') as f:
    for line in f.readlines():
        if line.startswith('>'):
            fasta_id_list.append(line.strip()[1:])
assert(len(fasta_id_list) == len(set(fasta_id_list)))
if set(fasta_id_list) == set(uniprot_id_list):
    print('identical')
