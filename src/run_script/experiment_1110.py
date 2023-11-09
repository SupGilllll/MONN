import os
os.chdir('/data/zhao/MONN/src')

# measures = ['KIKD']
# settings = ['new_new']
# thresholds = [0.3, 0.4, 0.5]
# out_path = "../results/1110/baseline"

# for measure in measures:
#     for setting in settings:
#         for threshold in thresholds:
#             os.system(f'python CPI_train.py {measure} {setting} {threshold} blosum62 &> {out_path}/{measure}_{setting}_{threshold}.log')

# out_path = "../results/1110/pure_transformer"
# for measure in measures:
#     for setting in settings:
#         for threshold in thresholds:
#             os.system(f'python transformer_train.py --measure {measure} --setting {setting} --clu_thre {threshold} &> {out_path}/{measure}_{setting}_{threshold}.log')

# out_path = "../results/1110/graph_transformer"
# for measure in measures:
#     for setting in settings:
#         for threshold in thresholds:
#             os.system(f'python transformer_train_graph.py --measure {measure} --setting {setting} --clu_thre {threshold} --lr 0.00085 --d_model 224 --nhead 1 --activation elu --optimizer RAdam --scheduler StepLR_10 &> {out_path}/{measure}_{setting}_{threshold}.log')

measures = ['KIKD']
settings = ['new_new']
thresholds = [0.3, 0.4, 0.5]
out_path = "../results/1110_2/baseline"

for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            os.system(f'python CPI_train.py {measure} {setting} {threshold} blosum62 &> {out_path}/{measure}_{setting}_{threshold}.log')

out_path = "../results/1110_2/pure_transformer"
for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            os.system(f'python transformer_train.py --measure {measure} --setting {setting} --clu_thre {threshold} &> {out_path}/{measure}_{setting}_{threshold}.log')

out_path = "../results/1110_2/graph_transformer"
for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            os.system(f'python transformer_train_graph.py --measure {measure} --setting {setting} --clu_thre {threshold} --lr 0.00085 --d_model 224 --nhead 1 --activation elu --optimizer RAdam --scheduler StepLR_10 &> {out_path}/{measure}_{setting}_{threshold}.log')
