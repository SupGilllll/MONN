import pickle
import os 
import numpy as np
os.chdir('/data/zhao/MONN/create_dataset')
# seen = set()
# dup = []
# with open('./out1.6_uniprot_uniprot_mapping.tab') as f:
#     for line in f.readlines():
#         pid = line.strip().split('\t')[0]
#         if pid in seen:
#             dup.append(pid)
#         else:
#             seen.add(pid)
# for pid in dup:
#     print(pid)

# uid_list = []
# uid_set = set()
# with open('./out1.6_uniprot_uniprot_mapping.tab') as f:
#     for line in f.readlines()[1:]:
#         pid = line.strip().split('\t')[1]
#         uid_list.append(pid)
#         uid_set.add(pid)
# print(len(uid_set), len(uid_list))
# with open('./out1.5_uniprotid_list.txt') as f:
#     for line in f.readlines():
#         pid = line.strip()
#         if pid not in uid_list:
#             print(pid)
# print(len(uid_list), len(set(uid_list)))
# cnt = 0
# with open('./out1.6_pdbbind_seqs.fasta') as f:
#     for line in f.readlines():
#         if line.startswith('>'):
#             cnt += 1
# print(cnt)

# protein_set  = set()
# ligand_set = set()
# with open('./out2_pdbbind_all_datafile.tsv', 'r') as f:
#     for line in f.readlines():
#         objects = line.strip().split('\t')
#         protein_set.add(objects[1])
#         ligand_set.add(objects[2])
# print(len(protein_set), len(ligand_set))

# ic50, kd, ki = 0, 0, 0
# with open('./out2_pdbbind_all_datafile.tsv', 'r') as f:
#     for line in f.readlines():
#         type = line.strip().split('\t')[5]
#         if type == 'Kd':
#             kd += 1
#         elif type == 'Ki':
#             ki += 1
#         elif type == 'IC50':
#             ic50 += 1
# print(ic50, kd, ki)

# f = open('./plip_results/output_'+'7gpb'+'.txt')
# isheader = False
# for line in f.readlines():
#     if line[0] == '*':
#         bond_type = line.strip().replace('*','')
#         isheader = True
#     if line[0] == '|':
#         if isheader:
#             header = line.replace(' ','').split('|')
#             print(header)
#             isheader = False
#             continue
#         lines = line.replace(' ','').split('|')
#         print(lines)

# ab = 55
# if 55 in [ab]:
#     print("you are right")

# from Bio.PDB import PDBParser, Selection
# from Bio.PDB.Polypeptide import three_to_one
# def get_seq(pdbid):
# 	p = PDBParser()
# 	structure = p.get_structure(pdbid, './pdb_files/'+pdbid+'.pdb')
# 	seq_dict = {}
# 	idx_to_aa_dict = {}
# 	for model in structure:
# 		for chain in model:
# 			chain_id = chain.get_id()
# 			if chain_id == ' ':
# 				continue
# 			seq = ''
# 			id_list = []
# 			for res in chain:
# 				if res.get_id()[0] != ' ' or res.get_id()[2] != ' ':   # remove HETATM
# 					continue
# 				print(res.get_id())
# 				# seq+=three_to_one(res.get_resname())
# 				# print(chain_id+str(res.get_id()[1])+res.get_id()[2].strip())
# 				# idx_to_aa_dict[chain_id+str(res.get_id()[1])+res.get_id()[2].strip()] = three_to_one(res.get_resname())
# 				# # try:
# 				# # 	seq+=three_to_one(res.get_resname())
# 				# # 	print(chain_id+str(res.get_id()[1])+res.get_id()[2].strip())
# 				# # 	idx_to_aa_dict[chain_id+str(res.get_id()[1])+res.get_id()[2].strip()] = three_to_one(res.get_resname())
# 				# # except:
# 				# # 	print('unexpected aa name', res.get_resname())
# 				# id_list.append(res.get_id()[1])
# 			seq_dict[chain_id] = (seq,id_list)
# 	return seq_dict, idx_to_aa_dict
# seq_dict, idx_dict = get_seq('5swg')
# print('happy')

# def get_bonds(pdbid, ligand, atom_idx_list):
# 	bond_list = []
# 	f = open('./plip_results/output_'+pdbid+'.txt')
# 	isheader = False
# 	for line in f.readlines():
# 		if line[0] == '*':
# 			bond_type = line.strip().replace('*','')
# 			isheader = True
# 		if line[0] == '|':
# 			if isheader:
# 				header = line.replace(' ','').split('|')
# 				isheader = False
# 				continue
# 			lines = line.replace(' ','').split('|')
# 			if ligand not in lines[5] or ligand not in lines[6]:
# 				continue
# 			if ligand in lines[5]:
# 				aa_id, aa_name, aa_chain, ligand_id, ligand_name, ligand_chain = int(lines[1]), lines[2], lines[3], int(lines[4]), lines[5], lines[6]
# 			elif ligand in lines[6]:
# 				aa_id, aa_name, aa_chain, ligand_id, ligand_name, ligand_chain = int(lines[1]), lines[2], lines[3], int(lines[5]), lines[6], lines[7]
# 			if bond_type in ['Hydrogen Bonds', 'Water Bridges'] :
# 				atom_idx1, atom_idx2 = int(lines[12]), int(lines[14])
# 				if atom_idx1 in atom_idx_list and atom_idx2 in atom_idx_list:   # discard ligand-ligand interaction
# 					continue
# 				if atom_idx1 in atom_idx_list:
# 					atom_idx_ligand, atom_idx_protein = atom_idx1, atom_idx2
# 				elif atom_idx2 in atom_idx_list:
# 					atom_idx_ligand, atom_idx_protein = atom_idx2, atom_idx1
# 				else:
# 					print(pdbid, ligand, bond_type, 'error: atom index in plip result not in atom_idx_list')
# 					print(atom_idx1, atom_idx2)
# 					return None
# 				bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein], ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))
# 			elif bond_type == 'Hydrophobic Interactions':
# 				atom_idx_ligand, atom_idx_protein = int(lines[8]), int(lines[9])
# 				if  atom_idx_ligand not in atom_idx_list: 
# 					continue
# 				elif atom_idx_ligand not in atom_idx_list:
# 					print('error: atom index in plip result not in atom_idx_list')
# 					print('Hydrophobic Interactions', atom_idx_ligand, atom_idx_protein)
# 					return None
# 				bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein], ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))
# 			elif bond_type in ['pi-Stacking', 'pi-Cation Interactions']:
# 				atom_idx_ligand_list = list(map(int, lines[12].split(',')))
# 				if len(set(atom_idx_ligand_list).intersection(set(atom_idx_list))) != len(atom_idx_ligand_list):
# 					print(bond_type, 'error: atom index in plip result not in atom_idx_list')
# 					print(atom_idx_ligand_list)
# 					return None
# 				bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [], ligand_chain, ligand_name, ligand_id, atom_idx_ligand_list))
# 			elif bond_type == 'Salt Bridges':
# 				atom_idx_ligand_list = list(set(map(int, lines[11].split(','))))
# 				if len(set(atom_idx_ligand_list).intersection(set(atom_idx_list))) != len(atom_idx_ligand_list):
# 					print('error: atom index in plip result not in atom_idx_list')
# 					print('Salt Bridges', atom_idx_ligand_list, set(atom_idx_ligand_list).intersection(set(atom_idx_list)))
# 					return None
# 				bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [], ligand_chain, ligand_name, ligand_id, atom_idx_ligand_list))
# 			elif bond_type == 'Halogen Bonds':
# 				atom_idx1, atom_idx2 = int(lines[11]), int(lines[13])
# 				if atom_idx1 in atom_idx_list and atom_idx2 in atom_idx_list:   # discard ligand-ligand interaction
# 					continue
# 				if atom_idx1 in atom_idx_list:
# 					atom_idx_ligand, atom_idx_protein = atom_idx1, atom_idx2
# 				elif atom_idx2 in atom_idx_list:
# 					atom_idx_ligand, atom_idx_protein = atom_idx2, atom_idx1
# 				else:
# 					print('error: atom index in plip result not in atom_idx_list')
# 					print('Halogen Bonds', atom_idx1, atom_idx2)
# 					return None
# 				bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein], ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))
# 			else:
# 				print('bond_type',bond_type)
# 				print(header)
# 				print(lines)
# 				return None
# 	f.close()
# 	if len(bond_list) != 0:
# 		return bond_list

# import os
# ligand_set = set()
# protein_set = set()
# with open('./out2_pdbbind_all_datafile.tsv', 'r') as f:
#     for line in f.readlines():
#         objects = line.strip().split('\t')
#         protein_set.add(objects[0])
#         ligand_set.add(objects[2])
# print(len(protein_set))
# print(len(ligand_set))
# for ligand in list(ligand_set):
#     if not os.path.isfile(f'./pdb_files/{ligand}_ideal.pdb') or os.stat(f'./pdb_files/{ligand}_ideal.pdb').st_size == 0:
#         print(ligand)
# for protein in list(protein_set):
#     if not os.path.isfile(f'./pdb_files/{protein}.pdb') or os.stat(f'./pdb_files/{protein}.pdb').st_size == 0:
#         print(protein)

# import pickle 
# import os
# os.chdir('/data/zhao/MONN/create_dataset')
# with open('./out4_interaction_dict', 'rb') as f:
#     dict = pickle.load(f)
# print('happy')
# with open('./out4_readable.txt', 'w') as f:
#     for line in list(dict):
#         f.write(line + '\n')

# import os
# os.chdir('/data/zhao/MONN/create_dataset')
# from Bio.PDB import PDBParser
# def get_mol_from_ligandpdb(ligand):
# 	if not os.path.exists('./pdb_files/'+ligand+'_ideal.pdb'):
# 		return None, None, None
# 	name_order_list = []
# 	name_to_idx_dict, name_to_element_dict = {}, {}
# 	p = PDBParser()
# 	structure = p.get_structure(ligand, './pdb_files/'+ligand+'_ideal.pdb')
# 	for model in structure:
# 		for chain in model:
# 			chain_id = chain.get_id()
# 			for res in chain:
# 				if ligand == res.get_resname():
# 					#print(ligand,res.get_resname(),res.get_full_id())
# 					for atom in res:
# 						name_order_list.append(atom.get_id())
# 						name_to_element_dict[atom.get_id()] = atom.element
# 						name_to_idx_dict[atom.get_id()] = atom.get_serial_number()-1
# 	#print('check', name_to_idx_dict.items())
# 	if len(name_to_idx_dict) == 0:
# 		return None, None, None
# 	return name_order_list, name_to_idx_dict, name_to_element_dict
# a, b, c = get_mol_from_ligandpdb('LXX')
# print('happy happy happy')

# import pickle 
# import os
# os.chdir('/data/zhao/MONN/create_dataset')
# with open('./out5_pocket_dict', 'rb') as f:
#     dict = pickle.load(f)
# print('happy happy happy')

# from Bio.PDB import PDBParser, Selection
# from Bio.PDB.Polypeptide import three_to_one
# from Bio import BiopythonWarning
# import warnings
# def get_seq_dict(pdbid, file_type):
# 	p = PDBParser()
# 	if os.path.exists('./pdbbind_files/'+pdbid+'/'+pdbid+'_'+file_type+'.pdb'):
# 		structure = p.get_structure(pdbid, './pdbbind_files/'+pdbid+'/'+pdbid+'_'+file_type+'.pdb')
# 	#elif os.path.exists('../pdbbind/refined-set/'+pdbid+'/'+pdbid+'_'+file_type+'.pdb'):
# 		#structure = p.get_structure(pdbid, '../pdbbind/refined-set/'+pdbid+'/'+pdbid+'_'+file_type+'.pdb')
# 	else:
# 		print(pdbid+' file not exist')
# 		return None
# 	seq_dict = {}
# 	for model in structure:
# 		for chain in model:
# 			chain_id = chain.get_id()
# 			if chain_id == ' ':
# 				continue
# 			seq = ''
# 			id_list = []
# 			for res in chain:
# 				if res.get_id()[0] != ' ':   # remove HETATM ?
# 					continue
# 				try:
# 					seq+=three_to_one(res.get_resname())
# 				except:
# 					print('unexpected aa name', res.get_resname())
# 				print(str(res.get_id()))
# 				print(str(res.get_id()[1]))
# 				print(str(res.get_id()[2]))
# 				id_list.append(str(res.get_id()[1])+str(res.get_id()[2]))
# 			seq_dict[chain_id] = (seq,id_list)
# 	return seq_dict
# dict = get_seq_dict('6mu1','pocket')

# def get_result_dict():
# 	result_dict = {}
# 	f = open('./smith-waterman-src/out6.3_pdb_align.txt')
# 	i = -1
# 	seq_target, seq_query, align = '', '', ''
# 	pdb_ratio_dict = {}
# 	for line in f.readlines():
# 		i += 1
# 		if i%4 == 0:
# 			if 'target_name' in line:
# 				if len(seq_target) != 0:
# 					result_dict[target_name] = (seq_target, seq_query, align, target_start, query_start)
# 				target_name = line.strip().split(' ')[-1]
# 				#print('target_name',target_name)
# 				seq_target, seq_query, align = '', '', ''
# 			else:
# 				seq_target += line.strip().split('\t')[1]
# 				#print('seq_target',seq_target)
# 		elif i%4 == 1:
# 			if 'query_name' in line:
# 				query_name = line.strip().split(' ')[-1]
# 				#print('query_name',query_name)
# 			else:
# 				align += line.strip('\n').split('\t')[1]
# 				#print('align',align)
# 		elif i%4 == 2:
# 			if 'optimal_alignment_score' in line:
# 				for item in line.strip().split('\t'):
# 					if item.split(' ')[0] == 'target_begin:':
# 						target_start = int(item.split(' ')[1])
# 					elif item.split(' ')[0] == 'query_begin:':
# 						query_start = int(item.split(' ')[1])
# 			else:
# 				seq_query += line.strip().split('\t')[1]
# 	f.close()
# 	return result_dict

# dict = get_result_dict()

# with open('./out7_final_pairwise_interaction_dict', 'rb') as f:
#     dict1 = pickle.load(f)
# with open('./out4_interaction_dict', 'rb') as f:
#     dict2 = pickle.load(f)
# print('happy happy happy')

# with open('./out5_pocket_dict', 'rb') as f:
#     dict1 = pickle.load(f)
# with open('./out8_final_pocket_dict', 'rb') as f:
#     dict2 = pickle.load(f)
# print('happy happy happy')

# remove_id_list = []
# not_in_dict7_list = []
# with open('./out7_final_pairwise_interaction_dict_backup', 'rb') as f:
#     dict7 = pickle.load(f)
# with open('./out8_final_pocket_dict_backup', 'rb') as f:
#     dict8 = pickle.load(f)
# print(len(dict7), len(dict8))
# dict7_list = list(dict7.keys())
# dict8_list = list(dict8.keys())
# for key in dict8:
#     if key not in dict7_list:
#         not_in_dict7_list.append(key)
#     elif len(dict8[key]['pocket_in_uniprot_seq']) == 0:
#         remove_id_list.append(key)
# remove_dict8_list = [item for item in dict8_list if item not in remove_id_list]
# print(len(remove_dict8_list))
# intersect_list = set(dict7_list).intersection(remove_dict8_list)
# print(len(intersect_list))
# for key in dict7_list:
#     if key not in intersect_list:
#         del dict7[key]
# for key in dict8_list:
#     if key not in intersect_list:
#         del dict8[key]
# print(len(dict7), len(dict8))
# with open('out7_final_pairwise_interaction_dict','wb') as f:
# 	pickle.dump(dict7, f)
# with open('out8_final_pocket_dict','wb') as f:
# 	pickle.dump(dict8, f)

# import matplotlib.pyplot as plt
# with open('./out7_final_pairwise_interaction_dict', 'rb') as f:
#     dict7 = pickle.load(f)
id_list = []
with open('./out2_pdbbind_all_datafile.tsv', 'r') as f:
    for line in f.readlines():
        objects = line.strip().split('\t')
        id_list.append(objects[0])
with open('./out8_final_pocket_dict', 'rb') as f:
    dict8 = pickle.load(f)
if '5mka' in dict8:
    print('happy1')
if '5mka' in id_list:
    print('happy2')

# dict7_list = list(dict7.keys())
# dict8_list = list(dict8.keys())
# remove_list = []
# for key in dict7:
#     if len(dict7[key]['interact_in_uniprot_seq']) == 0:
#         remove_list.append(key)
# print(len(remove_list))
# removed_dict7_list = [item for item in dict7_list if item not in remove_list]
# print(len(removed_dict7_list))
# intersect_list = set(dict8_list).intersection(removed_dict7_list)
# print(len(intersect_list))
# for key in dict7_list:
#     if key not in intersect_list:
#         del dict7[key]
# for key in dict8_list:
#     if key not in intersect_list:
#         del dict8[key]
# print(len(dict7), len(dict8))
# with open('out7_final_pairwise_interaction_dict','wb') as f:
# 	pickle.dump(dict7, f)
# with open('out8_final_pocket_dict','wb') as f:
# 	pickle.dump(dict8, f)

# total_interaction_sites, in_pocket_sites, pocket_sites = 0, 0, 0
# for key in dict7:
#     interact_list = dict7[key]['interact_in_uniprot_seq']
#     pocket_list = dict8[key]['pocket_in_uniprot_seq']
#     total_interaction_sites += len(interact_list)
#     pocket_sites += len(pocket_list)
#     in_pocket_sites += len(set(interact_list).intersection(pocket_list))
# print('total_interaction_sites', total_interaction_sites)
# print('in_pocket_sites', in_pocket_sites)
# print('pocket_sites', pocket_sites)

# percentage_list = []
# p = 0
# cnt = len(dict7)
# for key in dict7:
#     interact_list = dict7[key]['interact_in_uniprot_seq']
#     pocket_list = dict8[key]['pocket_in_uniprot_seq']
#     in_pocket_sites = len(set(interact_list).intersection(pocket_list))
#     percentage_list.append((in_pocket_sites/len(interact_list))*100)
#     p += (in_pocket_sites/len(interact_list)) * 100 / cnt
# print(p)

# import numpy as np
# # define the ranges you're interested in as a list of tuples
# ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]

# # use numpy to count the number of elements in each range
# counts, _ = np.histogram(percentage_list, bins=[r[0] for r in ranges] + [r[1] for r in ranges[-1:]])

# # calculate the percentage of elements in each range
# total = len(percentage_list)
# percentages = [100 * count / total for count in counts]

# # print the percentages
# for i, r in enumerate(ranges):
#     print(f"{r}: {percentages[i]:.2f}%")

# uniprotids_list = np.load('pdbbind_protein_list.npy')
# mat = np.load('pdbbind_protein_sim_mat.npy')
# print(np.dtype(mat[0,1]))
