import numpy as np
import os
os.chdir('/data/zhao/MONN/src')

ori_protein_list = np.load('../data_Archive/pdbbind_protein_list.npy').tolist()
protein_sim_mat = np.load('../data_Archive/pdbbind_protein_sim_mat.npy').astype(np.float32)
print(ori_protein_list[0])
print('happy')