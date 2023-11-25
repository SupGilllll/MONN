import pickle
import numpy as np

int_types = ['Hydrogen Bonds', 'Water Bridges', 'Hydrophobic Interactions', 'pi-Stacking', 'pi-Cation Interactions', 'Salt Bridges', 'Halogen Bonds']
map_dict = {int_types[i]: i + 1 for i in range(len(int_types))}
with open('/data/zhao/MONN/create_dataset/out7_final_pairwise_interaction_dict','rb') as f:
    interaction_dict = pickle.load(f)

pdbid_list = []
non_count = 0
int_count = 0
multi_count = 0
total_count = 0
int_dict = dict()
for pdbid in interaction_dict.keys():
    atom_element = np.array(interaction_dict[pdbid]['atom_element'], dtype=str)
    atom_name_list = np.array(interaction_dict[pdbid]['atom_name'], dtype=str)
    nonH_position = np.where(atom_element != ('H'))[0]
    atom_name_list = atom_name_list[nonH_position].tolist()
    pairwise_mat = np.zeros((len(nonH_position), len(interaction_dict[pdbid]['uniprot_seq']), 8), dtype=np.int32)
    pairwise_mat[:, :, 0] = 1
    for atom_name, bond_type in interaction_dict[pdbid]['atom_bond_type']:
        real_type = bond_type.split('_')[0]
        atom_idx = atom_name_list.index(str(atom_name))
        assert atom_idx < len(nonH_position)
        for seq_idx, bond_type_seq in interaction_dict[pdbid]['residue_bond_type']:
            if bond_type == bond_type_seq:
                if pairwise_mat[atom_idx, seq_idx, 0] == 1:
                    pairwise_mat[atom_idx, seq_idx, 0] = 0
                pairwise_mat[atom_idx, seq_idx, map_dict[real_type]] = 1
    sum_array = np.sum(pairwise_mat, axis = -1)
    non_count += np.sum(pairwise_mat[:, :, 0] == 1)
    int_count += np.sum(pairwise_mat[:, :, 0] != 1)
    count_mul = np.sum(sum_array >= 2)
    total_count += pairwise_mat.shape[0] * pairwise_mat.shape[1]
    positions_greater_than_2 = np.argwhere(pairwise_mat[:, :, 0] != 1)
    for position in positions_greater_than_2:
        string = ''
        for idx, element in enumerate(pairwise_mat[position[0], position[1]]):
            if element == 1:
                if idx == 0:
                    string += 'none_'
                else:
                    string += int_types[idx - 1] + '_'
        key_str = string[:-1]
        if key_str not in int_dict:
            int_dict[key_str] = 1
        else:
            int_dict[key_str] += 1
    if count_mul > 0:
        multi_count += count_mul
        pdbid_list.append(pdbid)
int_dict["Non-interaction"] = non_count
print(len(pdbid_list))
print(non_count, int_count, multi_count, total_count)
print(int_dict)