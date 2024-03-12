# import pickle 

# with open('/data/zhao/MONN/create_dataset/out4_interaction_dict', 'rb') as f:
#     d4 = pickle.load(f)
# with open('/data/zhao/MONN/create_dataset/out7_final_pairwise_interaction_dict', 'rb') as f:
#     d7 = pickle.load(f)
# with open('/data/zhao/MONN/create_dataset/out8_final_pocket_dict', 'rb') as f:
#     d8 = pickle.load(f)
# print('happy')

import numpy as np
import time

# Define the array to be appended
to_append = np.array([1, 2, 3])
n_appends = 100000  # Number of times to append

# Measure time for numpy.append in a loop
start_time = time.time()
result_append = np.array([])
for _ in range(n_appends):
    result_append = np.append(result_append, to_append)
time_append = time.time() - start_time

# Measure time for list accumulation and single concatenation
start_time = time.time()
# accumulated_list = [to_append for _ in range(n_appends)]
accumulated_list = []
for _ in range(n_appends):
    accumulated_list.append(to_append)
result_list = np.concatenate(accumulated_list)
time_list = time.time() - start_time

print(time_append, time_list)

