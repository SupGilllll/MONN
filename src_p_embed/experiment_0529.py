import os
os.chdir('/data/zhao/MONN/src_p_embed')

output_path = '../results/0529_modified'
measures = ['KIKD', 'IC50']
settings = ['new_new', 'new_protein', 'new_compound']
thresholds = [0.3, 0.4, 0.5]

for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            os.system(f'python CPI_train.py {measure} {setting} {threshold} > {output_path}/{measure}_{setting}_{threshold}.log')
