# Hyper Parameters
# learning rate || hidden_dim || heads of multi-attention || layers of encoder / decoder
# activation function || learning rate optimizer / scheduler
import os
os.chdir('/data/zhao/MONN/src')

measures = ['KIKD']
settings = ['new_protein']
thresholds = [0.3, 0.4, 0.5]
out_path = "../results/0710/blosum62"

for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            os.system(f'python transformer_train.py --clu_thre {threshold} &> {out_path}/{measure}_{setting}_{threshold}.log')