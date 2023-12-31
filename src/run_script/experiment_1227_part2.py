import os
os.chdir('/data/zhao/MONN/src')


measures = ['ALL']
settings = ['new_new']
thresholds = [0.4, 0.3, 0.5]
out_path = "../results/1227/binary-class"

for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            os.system(f'python transformer_train_graph.py --measure {measure} --setting {setting} --clu_thre {threshold} &> {out_path}/useless.log')
