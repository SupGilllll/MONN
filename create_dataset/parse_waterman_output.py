import os
import numpy as np
os.chdir('/data/zhao/MONN/create_dataset')

uniprotids_list = np.load('pdbbind_protein_list.npy').tolist()
dimension = len(uniprotids_list)
sim_mat = np.zeros((dimension, dimension))
for row_idx, uid in enumerate(uniprotids_list):
    if row_idx % 100 == 0:
        print(f'{row_idx} samples have been processed')
    col_idx = 0
    with open(f'./uniprot_fasta_output/output_{uid}.txt', 'r') as f:
        for line in f.readlines():
            if line.startswith('optimal_alignment_score'):
                sim_mat[row_idx, col_idx] = int(line.strip().split('\t')[0].split()[1])
                col_idx += 1

normailzed_mat = np.zeros((dimension, dimension))
for row in range(dimension):
    for col in range(dimension):
        normailzed_mat[row, col] = sim_mat[row, col] / np.sqrt(sim_mat[row, row] * sim_mat[col, col])

np.save('pdbbind_protein_sim_mat.npy', normailzed_mat)
