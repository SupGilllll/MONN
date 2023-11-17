import os
os.chdir('/data/zhao/MONN/src')

measure = 'KIKD'
setting = 'new_new'
thresholds = [0.3, 0.4, 0.5]
weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
out_path = "../results/1120/loss_weight"

for threshold in thresholds:
    for weight in weights:
        os.system(f'python transformer_train_graph.py --clu_thre {threshold} --loss_weight {weight} --lr 0.00085 --d_model 224 --scheduler StepLR_10 &> {out_path}/{measure}_{setting}_{threshold}_{weight}.log')
