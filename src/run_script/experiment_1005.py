import os
os.chdir('/data/zhao/MONN/src')

# measures = ['KIKD']
# settings = ['new_compound', 'new_new']
# thresholds = [0.3, 0.4, 0.5]
# out_path = "../results/1005/baseline"

# for measure in measures:
#     for setting in settings:
#         for threshold in thresholds:
#             os.system(f'python CPI_train.py {measure} {setting} {threshold} blosum62 &> {out_path}/{measure}_{setting}_{threshold}.log')

measures = ['IC50']
settings = ['new_protein', 'new_compound', 'new_new']
thresholds = [0.3, 0.4, 0.5]
out_path = "../results/1005/baseline"

for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            os.system(f'python CPI_train.py {measure} {setting} {threshold} blosum62 &> {out_path}/{measure}_{setting}_{threshold}.log')


# out_path = "../results/0724/transformer_base"
# for measure in measures:
#     for setting in settings:
#         for threshold in thresholds:
#             os.system(f'python transformer_train.py --clu_thre {threshold} --epochs 50 --cuda_device 0 &> {out_path}/{measure}_{setting}_{threshold}.log')

