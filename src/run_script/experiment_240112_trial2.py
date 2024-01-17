import os
os.chdir('/data/zhao/MONN/src')


measures = ['ALL']
settings = ['new_new']
thresholds = [0.3, 0.4, 0.5]
out_path = "../results/240116/trial2"
for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            os.system(f'python transformer_train_graph.py --measure {measure} --setting {setting} --clu_thre {threshold} --cuda_device 1 &> {out_path}/{measure}_{setting}_{threshold}.log')

# out_path = "../results/240112/trial5"
# for measure in measures:
#     for setting in settings:
#         for threshold in thresholds:
#             os.system(f'python transformer_train_graph.py --measure {measure} --setting {setting} --clu_thre {threshold} --cuda_device 1 --decoder_layers 5 &> {out_path}/{measure}_{setting}_{threshold}.log')

# out_path = "../results/240112/trial6"
# for measure in measures:
#     for setting in settings:
#         for threshold in thresholds:
#             os.system(f'python transformer_train_graph.py --measure {measure} --setting {setting} --clu_thre {threshold} --cuda_device 1 --decoder_layers 6 &> {out_path}/{measure}_{setting}_{threshold}.log')


