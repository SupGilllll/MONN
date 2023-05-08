import numpy as np
import os
os.chdir('/data/zhao/MONN/create_dataset')

# def get_fasta_dict():
# 	uniprot_dict = {}
# 	name,seq = '',''
# 	with open('out1.6_pdbbind_seqs.fasta') as f:
# 		for line in f.readlines():
# 			if line[0] == '>':
# 				if name != '':
# 					uniprot_dict[name] = seq
# 				name = line.split('|')[1]
# 				seq = ''
# 			else:
# 				seq += line.strip()
# 		uniprot_dict[name] = seq
# 	print('uniprot_dict step1',len(uniprot_dict))
# 	return uniprot_dict

# uniprot_id_list = np.load('pdbbind_protein_list.npy').tolist()
def get_uniprotid_to_seq():
    # seq_dict = get_fasta_dict()
    uniprotid_to_seq = {}
    with open('out2_pdbbind_all_datafile.tsv') as f:
        for line in f.readlines():
            _, uniprotid, _, _, seq, _, _ = line.strip().split('\t')
            # assert seq_dict[uniprotid] == seq
            uniprotid_to_seq[uniprotid] = seq
    print('uniprotid_to_seq',len(uniprotid_to_seq))
    return uniprotid_to_seq
uniprotid_to_seq = get_uniprotid_to_seq()

# id_nums = len(uniprotid_to_seq)
# with open('./uniprot_fasta/target_uniprot_all.fasta', 'w') as target_f:
#     for uniproid, seq in uniprotid_to_seq.items():
#         target_f.write(f'>{uniproid}\n')
#         target_f.write(f'{seq}\n')
#         with open(f'./uniprot_fasta/query_{uniproid}.fasta', 'w') as query_f:
#             for i in range(id_nums):
#                 query_f.write(f'>{uniproid}\n')
#                 query_f.write(f'{seq}\n')
