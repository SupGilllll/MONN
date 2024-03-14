import numpy as np

kikd = np.load('../preprocessing/wlnn_train_list_KIKD.npy').astype(str).tolist()
ic50 = np.load('../preprocessing/wlnn_train_list_IC50.npy').astype(str).tolist()
combined = np.load('../preprocessing/wlnn_train_list_ALL.npy').astype(str).tolist()

count1 = 0
count2 = 0
for sample in combined:
    if sample in kikd:
        count1 += 1
    elif sample in ic50:
        count2 += 1
print(len(kikd), len(ic50))
print(count1, count2, len(combined))

pid_dict = {}
cid_dict = {}
measure_dict = {}
with open('../create_dataset/out2_pdbbind_all_datafile.tsv', 'r') as f:
    for line in f.readlines():
        pdbid, pid, cid, _, _, measure, _  = line.strip().split('\t')
        pdbid = str(pdbid)
        measure = str(measure)
        pid_dict[pdbid] = pid
        cid_dict[pdbid] = cid
        measure_dict[pdbid] = measure

protein_set = set()
compound_set = set()
ki, kd, ic = 0, 0, 0
for samples in kikd:
    if measure_dict[samples] == 'Kd':
        kd += 1
    elif measure_dict[samples] == 'Ki':
        ki += 1
    elif measure_dict[samples] == 'IC50':
        ic += 1
    protein_set.add(pid_dict[samples])
    compound_set.add(cid_dict[samples])

print(ki, kd, ic)
print(len(protein_set))
print(len(compound_set))
# common_elements = np.intersect1d(kikd, combined)
# print(len(common_elements))
# common_elements = np.intersect1d(ic50, combined)
# print(len(common_elements))
# print(len(kikd) != len(set(kikd)))
# print(len(ic50) != len(set(ic50)))
# print(len(combined) != len(set(combined)))
# count_k = 0
# count_c = 0
print(len(kikd), len(ic50), len(combined))