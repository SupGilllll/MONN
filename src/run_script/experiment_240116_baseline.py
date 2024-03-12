import os
os.chdir('/data/zhao/MONN/src')

measures = ['ALL']
settings = ['new_new']
thresholds = [0.3, 0.4, 0.5]
out_path = "../results/240116/baseline/pocket_region"

for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            os.system(f'python CPI_train.py {measure} {setting} {threshold} blosum62 &> {out_path}/{measure}_{setting}_{threshold}.log')

out_path = "../results/240116/transformer_trial2/pocket_region"

for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            os.system(f'python transformer_train_graph.py --measure {measure} --setting {setting} --clu_thre {threshold} --cuda_device 1 --epochs 30 &> {out_path}/{measure}_{setting}_{threshold}.log')
