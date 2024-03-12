import os
os.chdir('/data/zhao/MONN/src')

measures = ['ALL']
settings = ['new_new']
thresholds = [0.3, 0.4, 0.5]
weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
out_path = "../results/240116/transformer_trial2/loss_weight"
for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            for weight in weights:
                os.system(f'python transformer_train_graph.py --measure {measure} --setting {setting} --clu_thre {threshold} --loss_weight {weight} --cuda_device 1 --epochs 30 &> {out_path}/{measure}_{setting}_{threshold}_{weight}_epochs30.log')
