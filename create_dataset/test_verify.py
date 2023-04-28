import os 
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
# 	structure = p.get_structure(pdbid, './'+pdbid+'.pdb')
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
# 				try:
# 					seq+=three_to_one(res.get_resname())
# 					print(chain_id+str(res.get_id()[1])+res.get_id()[2].strip())
# 					idx_to_aa_dict[chain_id+str(res.get_id()[1])+res.get_id()[2].strip()] = three_to_one(res.get_resname())
# 				except:
# 					print('unexpected aa name', res.get_resname())
# 				id_list.append(res.get_id()[1])
# 			seq_dict[chain_id] = (seq,id_list)
# 	return seq_dict, idx_to_aa_dict
# get_seq('6o2p')

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

import pickle 
import os
os.chdir('/data/zhao/MONN/create_dataset')
with open('./out5_pocket_dict', 'rb') as f:
    dict = pickle.load(f)
print('happy happy happy')

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
