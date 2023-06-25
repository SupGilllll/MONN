import os
os.chdir('/data/zhao/MONN/src_p_embed')

# output_path = '../results/0612/esm'
# measures = ['KIKD']
# settings = ['new_protein']
# thresholds = [0.3, 0.4, 0.5]

# for measure in measures:
#     for setting in settings:
#         for threshold in thresholds:
#             os.system(f'python CPI_train.py {measure} {setting} {threshold} > {output_path}/{measure}_{setting}_{threshold}.log')

measures = ['KIKD']
settings = ['new_protein']
thresholds = [0.3, 0.4, 0.5]
embed_models = ['t33']
output_path = '../results/0612'

for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            for embed_model in embed_models:
                os.system(f'python CPI_train.py {measure} {setting} {threshold} {embed_model} > {output_path}/SurfaceAUC_{embed_model}/{measure}_{setting}_{threshold}.log')
