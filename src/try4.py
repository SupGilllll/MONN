import pickle 

with open('/data/zhao/MONN/create_dataset/out4_interaction_dict', 'rb') as f:
    d4 = pickle.load(f)
with open('/data/zhao/MONN/create_dataset/out7_final_pairwise_interaction_dict', 'rb') as f:
    d7 = pickle.load(f)
with open('/data/zhao/MONN/create_dataset/out8_final_pocket_dict', 'rb') as f:
    d8 = pickle.load(f)
print('happy')