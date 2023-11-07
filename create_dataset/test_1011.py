# with open('5k4i.dssp', 'r') as f:
#     lines = f.readlines()
# seq = ""
# p = 0
# while lines[p].strip()[0] != "#":
#     p += 1
# for i in range(p + 1, len(lines)):
#     aa = lines[i][13]
#     if aa == "!" or aa == "*":
#         continue
#     seq += aa
# print(len(seq))

import pickle
with open('./create_dataset/out4_interaction_dict', 'rb') as f:
    data1 = pickle.load(f)
# with open('/data/zhao/MONN/preprocessing/surface_area_dict', 'rb') as f:
#     data1 = pickle.load(f)
with open('/data/zhao/MONN/create_dataset/out5_pocket_dict', 'rb') as f:
    data2 = pickle.load(f)
print("load finish")

# import os

# with open('./out1.2_pdbbind_wget_complex.txt', 'r') as f:
#     for line in f.readlines()[:12]:
#         file_path = f'./pdb_files/{line.strip()[-8:]}'
#         print(file_path)
#         if file_path == './pdb_files/2ly0.pdb':
#             print("yes")
#         if not os.path.exists(file_path):
#             print(f"file {file_path} not in")

# file_path = './pdb_files/6ozp.pdb'
# if not os.path.exists(file_path):
#     print(f"file {file_path} not in")
