import os
os.chdir('/data/zhao/MONN/src')

measures = ['KIKD']
settings = ['new_new']
thresholds = [0.3, 0.4, 0.5]
out_path = "../results/1005/pure_transformer"

for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            os.system(f'python transformer_train.py --measure {measure} --setting {setting} --clu_thre {threshold} &> {out_path}/{measure}_{setting}_{threshold}.log')

