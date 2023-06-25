import json
import os
import pickle
import statistics
os.chdir('/data/zhao/MONN/src')

with open('./pid_len_dict', 'rb') as f:
    len_dict = pickle.load(f)
with open('../preprocessing/RSA.json', 'r') as f:
    data = json.load(f)
surface_area_dict = {}
percent_list = []
for sample in data:
    idx_list = []
    assert len_dict[sample['desc']] == len(sample['seq'])
    for idx, val in enumerate(sample['rsa']):
        if val > 0.20:
            idx_list.append(idx)
    surface_area_dict[sample['desc']] = idx_list
    percent_list.append(len(idx_list) / len_dict[sample['desc']])
with open('../preprocessing/surface_area_dict', 'wb') as f:
    pickle.dump(surface_area_dict, f)
print(statistics.mean(percent_list))
