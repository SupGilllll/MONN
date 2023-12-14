import os
os.chdir('/data/zhao/MONN/src')


measures = ['KIKD']
settings = ['new_new', 'new_protein', 'new_compound']
thresholds = [0.3, 0.4, 0.5]
# out_path = "../results/1204/graph_transformer"

# for measure in measures:
#     for setting in settings:
#         for threshold in thresholds:
#             if setting == 'new_new':
#                 os.system(f'python transformer_train_graph.py --measure {measure} --setting {setting} --clu_thre {threshold} &> {out_path}/{measure}_{setting}_{threshold}.log')
#             else:
#                 os.system(f'python transformer_train_graph.py --measure {measure} --setting {setting} --clu_thre {threshold} --epochs 30 &> {out_path}/{measure}_{setting}_{threshold}.log')

out_path = "../results/1204/graph_transformer/epoch15"
for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            if setting != 'new_new':
                os.system(f'python transformer_train_graph.py --measure {measure} --setting {setting} --clu_thre {threshold} &> {out_path}/{measure}_{setting}_{threshold}.log')
