# generate surface_dict
import pickle
import os

maxASA = {'A' : 113.0, 'R' : 241.0, 'N' : 158.0, 'D' : 151.0,
          'C' : 140.0, 'E' : 183.0, 'Q' : 189.0, 'G' : 85.0,
          'H' : 194.0, 'I' : 182.0, 'L' : 180.0, 'K' : 211.0,
          'M' : 204.0, 'F' : 218.0, 'P' : 143.0, 'S' : 122.0,
          'T' : 146.0, 'W' : 259.0, 'Y' : 229.0, 'V' : 160.0}

with open('../out4_interaction_dict', 'rb') as f:
    int_info = pickle.load(f)
with open('../out5_pocket_dict', 'rb') as f:
    data2 = pickle.load(f)

def parse_dssp(sample_id):
    pid = sample_id.split('_')[0]
    sample = int_info[sample_id]['sequence']
    with open(f'../dssp_files/{pid}.dssp', "r") as f:
        lines = f.readlines()
    p = 0
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        aa = lines[i][13]
        if aa == "!" or aa == "*" or aa == 'X':
            continue
        if aa.islower():
            aa = 'C'
        chain = lines[i][11]
        try:
            idx = int(lines[i][6:11].strip())
        except:
            continue
        if chain not in sample or idx not in sample[chain][1]:
            continue
        acc = float(lines[i][35:39].strip())
        rsa = acc / maxASA[aa]
        # print(rsa)
        if rsa >= 0.25:
            if sample_id not in surface_dict:
                surface_dict[sample_id] = dict()
                surface_dict[sample_id]['protein'] = int_info[sample_id]['sequence']
                surface_dict[sample_id]['surface'] = dict()
            if chain not in surface_dict[sample_id]['surface']:
                surface_dict[sample_id]['surface'][chain] = ['', []]
            surface_dict[sample_id]['surface'][chain][0] += aa
            surface_dict[sample_id]['surface'][chain][1].append(idx)

surface_dict = {}
i = 1
for sample_id in int_info.keys():
    if i % 100 == 0:
        print(f"------------process {i} samples------------")
    parse_dssp(sample_id)
    i += 1

for sample_id in surface_dict.keys():
    for chain in surface_dict[sample_id]['surface'].keys():
        surface_dict[sample_id]['surface'][chain] = (surface_dict[sample_id]['surface'][chain][0], 
                                                     surface_dict[sample_id]['surface'][chain][1])

with open('../out3_surface_dict', 'wb') as f:
    pickle.dump(surface_dict, f)
# parse_dssp('1uys_H1L')

print("------------finish processing!------------")