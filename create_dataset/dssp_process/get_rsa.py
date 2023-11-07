import pickle
import os
os.chdir('/data/zhao/MONN/create_dataset')

maxASA = {'A' : 113.0, 'R' : 241.0, 'N' : 158.0, 'D' : 151.0,
          'C' : 140.0, 'E' : 183.0, 'Q' : 189.0, 'G' : 85.0,
          'H' : 194.0, 'I' : 182.0, 'L' : 180.0, 'K' : 211.0,
          'M' : 204.0, 'F' : 218.0, 'P' : 143.0, 'S' : 122.0,
          'T' : 146.0, 'W' : 259.0, 'Y' : 229.0, 'V' : 160.0}

with open('./out4_interaction_dict', 'rb') as f:
    int_info = pickle.load(f)

def get_pdbid_to_uniprot():
    pdbid_to_uniprot = {}
    with open('./out2_pdbbind_all_datafile.tsv', 'r') as f:
        for line in f.readlines():
            pdbid, uniprotid, _, _, _, _, _ = line.strip().split('\t')
            pdbid_to_uniprot[pdbid] = uniprotid
    print('pdbid_to_uniprot',len(pdbid_to_uniprot))
    return pdbid_to_uniprot
pdbid_to_uniprot = get_pdbid_to_uniprot()

def parse_dssp(sample_id):
    pid = sample_id.split('_')[0]
    uid = pdbid_to_uniprot[pid]
    sample = int_info[sample_id]['sequence']
    with open(f'./dssp_files/{pid}.dssp', "r") as f:
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
            idx = int(lines[i][7:11].strip())
        except:
            continue
        if chain not in sample or idx not in sample[chain][1]:
            continue
        acc = float(lines[i][35:39].strip())
        rsa = acc / maxASA[aa]
        print(rsa)
        if rsa >= 0.25:
            if uid not in ordered_set:
                ordered_set[uid] = dict()
            if chain not in ordered_set[uid]:
                ordered_set[uid][chain] = set()
            ordered_set[uid][chain].add(idx)

ordered_set = {}
i = 0
for sample_id in list(int_info.keys())[:1]:
    if i % 100 == 0:
        print(f"------------process {i} samples------------")
    parse_dssp(sample_id)
    i += 1

for uid, seq in ordered_set.items():
    for chain, idx_list in ordered_set[uid].items():
        ordered_set[uid][chain] = sorted(ordered_set[uid][chain])
print("------------finish processing!------------")