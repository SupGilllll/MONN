import pickle

# with open('/data/zhao/MONN/create_dataset/out7_final_pairwise_interaction_dict', 'rb') as f:
#     interaction_dict = pickle.load(f)
# int_types = set()
# for value in interaction_dict.values():
#     for bond_type in value['residue_bond_type']:
#         int_type = bond_type[1].split('_')[0]
#         int_types.add(int_type)
# print(len(int_types), " types")
# for type in int_types:
#     print(type)

with open('/data/zhao/MONN/create_dataset/out4_interaction_dict', 'rb') as f:
    interaction_dict = pickle.load(f)
int_types = set()
for value in interaction_dict.values():
    for bond_type in value['residue_interact']:
        int_type = bond_type[2].split('_')[0]
        int_types.add(int_type)
print(len(int_types), " types")
for type in int_types:
    print(type)