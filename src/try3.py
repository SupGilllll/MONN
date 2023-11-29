import pickle

with open('/data/zhao/MONN/preprocessing_multiclass/pdbbind_all_combined_input_ALL', 'rb') as f:
        data = pickle.load(f)
print('happy')