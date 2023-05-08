import numpy as np

uniprot_id_list = []
id_set = set()
with open('./out2_pdbbind_all_datafile.tsv', 'r') as f:
    for line in f.readlines():
        uniprot_id = line.strip().split('\t')[1]
        if uniprot_id not in id_set:
            id_set.add(uniprot_id)
            uniprot_id_list.append(uniprot_id)

uniprot_id_array = np.array(uniprot_id_list, dtype='str')
np.save('pdbbind_protein_list.npy', uniprot_id_array)