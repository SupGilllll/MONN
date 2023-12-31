import os
os.chdir('/data/zhao/MONN/src')

measures = ['ALL']
settings = ['new_new']
thresholds = [0.4, 0.3, 0.5]
out_path = "../results/1227/baseline"

for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            os.system(f'python CPI_train.py {measure} {setting} {threshold} blosum62 &> {out_path}/useless.log')
