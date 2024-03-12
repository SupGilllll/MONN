import os
os.chdir('/data/zhao/MONN/src')


measures = ['ALL']
settings = ['new_new']
thresholds = [0.3, 0.4, 0.5]

out_path = "../results/240116/transformer/binary_binding"
for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            os.system(f'python transformer_train_binary_binding.py --measure {measure} --setting {setting} --clu_thre {threshold} --epochs 30 &> {out_path}/{measure}_{setting}_{threshold}_weight.log')

out_path = "../results/240116/transformer/binary_interaction"
for measure in measures:
    for setting in settings:
        for threshold in thresholds:
            os.system(f'python transformer_train_binary_interaction.py --measure {measure} --setting {setting} --clu_thre {threshold} --epochs 30 &> {out_path}/{measure}_{setting}_{threshold}_weight.log')
