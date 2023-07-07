import os
os.chdir('/data/zhao/MONN/src')

# measures = ['KIKD', 'IC50']
# settings = ['new_new', 'new_protein', 'new_compound']
# thresholds = [0.3, 0.4, 0.5, 0.6]
# measures = ['KIKD']
# settings = ['new_protein']
# thresholds = [0.3, 0.4, 0.5]
# embedding_method = ['blosum62', 'one-hot']
# out_path = "../results/0612"

# for measure in measures:
#     for setting in settings:
#         for threshold in thresholds:
#             for prot_embed in embedding_method:
#                 os.system(f'python CPI_train.py {measure} {setting} {threshold} {prot_embed} > {out_path}/{prot_embed}/{measure}_{setting}_{threshold}.log')

# measures = ['KIKD']
# settings = ['new_protein']
# thresholds = [0.3, 0.4, 0.5]
# embedding_method = ['blosum62']
# out_path = "../results/0612"

# for measure in measures:
#     for setting in settings:
#         for threshold in thresholds:
#             for prot_embed in embedding_method:
#                 os.system(f'python CPI_train.py {measure} {setting} {threshold} {prot_embed} > {out_path}/SurfaceAUC_{prot_embed}/{measure}_{setting}_{threshold}.log')

measures = ['KIKD']
settings = ['new_protein']
thresholds = [0.3]
embedding_method = ['blosum62']
out_path = "../results/0626"

for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            for prot_embed in embedding_method:
                os.system(f'python transformer_train.py {measure} {setting} {threshold} {prot_embed} > {out_path}/{prot_embed}/{measure}_{setting}_{threshold}.log')