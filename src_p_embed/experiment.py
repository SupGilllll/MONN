import os
os.chdir('/data/zhao/MONN/src_p_embed')

measures = ['KIKD', 'IC50']
settings = ['new_new', 'new_protein', 'new_compound']
thresholds = [0.3, 0.4, 0.5, 0.6]

for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            os.system(f'python CPI_train.py {measure} {setting} {threshold} > {measure}_{setting}_{threshold}.log')
