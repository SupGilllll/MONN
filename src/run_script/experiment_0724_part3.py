import os
os.chdir('/data/zhao/MONN/src')

measures = ['KIKD']
settings = ['new_protein']
thresholds = [0.3, 0.4, 0.5]
layers = [1, 2]
out_path = "../results/0724/transformer_novel1"

for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            for layer in layers:
                os.system(f'python transformer_train_novel.py --clu_thre {threshold} --num_layers {layer} --pos_encoding none1 \
                          &> {out_path}/{measure}_{setting}_{threshold}_{layer}.log')
                
out_path = "../results/0724/transformer_novel"

for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            for layer in layers:
                os.system(f'python transformer_train_novel.py --clu_thre {threshold} --num_layers {layer} \
                          &> {out_path}/{measure}_{setting}_{threshold}_{layer}.log')

                


