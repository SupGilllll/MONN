import os
os.chdir('/data/zhao/MONN/src')

# measures = ['KIKD', 'IC50']
# settings = ['new_new', 'new_protein', 'new_compound']
# thresholds = [0.3, 0.4, 0.5, 0.6]
measures = ['KIKD']
settings = ['new_new']
thresholds = [0.3, 0.4, 0.5]
out_path = "../results/0529_baseline"

for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            os.system(f'python CPI_train.py {measure} {setting} {threshold} > {out_path}/{measure}_{setting}_{threshold}.log')
