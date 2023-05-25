from collections import defaultdict
import numpy as np
import torch
import os
os.chdir('/data/zhao/MONN/src')
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

data = torch.load('./test_emb_esm2/1.pt')
print('happy')