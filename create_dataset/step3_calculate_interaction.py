import os
import time

pdbid_list = []
with open('./out2_pdbbind_all_datafile.tsv', 'r') as f:
    for line in f.readlines():
        pdbid = line.strip().split('\t')[0]
        pdbid_list.append(pdbid)
start = time.time()
for pdbid in pdbid_list:
    os.system(f'plip -f ./pdb_files/{pdbid}.pdb -q -t --name output_{pdbid} -o ./plip_results')
end = time.time()
print(end - start)