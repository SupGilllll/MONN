import os
os.chdir('/data/zhao/MONN/src')

measures = ['KIKD']
settings = ['new_new', 'new_protein', 'new_compound']
thresholds = [0.3, 0.4, 0.5]
out_path = "../results/1204/baseline"

for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            os.system(f'python CPI_train.py {measure} {setting} {threshold} blosum62 &> {out_path}/{measure}_{setting}_{threshold}.log')