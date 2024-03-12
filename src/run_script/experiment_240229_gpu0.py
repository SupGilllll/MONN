import os
os.chdir('/data/zhao/MONN/src')

thresholds = [0.3, 0.4, 0.5]
seeds = [84, 126]

for seed in seeds:
    out_path = "../results/240229/multi_task"
    for threshold in thresholds:
        os.system(f'python transformer_train_graph.py --clu_thre {threshold} --random_seed {seed} --cuda_device 0 &> {out_path}/{threshold}_seed{seed}.log')

    out_path = "../results/240229/binary_binding"
    for threshold in thresholds:
        os.system(f'python transformer_train_binary_binding.py --clu_thre {threshold} --random_seed {seed} --cuda_device 0 &> {out_path}/{threshold}_seed{seed}.log')

    out_path = "../results/240229/binary_interaction"
    for threshold in thresholds:
        os.system(f'python transformer_train_binary_interaction.py --clu_thre {threshold} --random_seed {seed} --cuda_device 0 &> {out_path}/{threshold}_seed{seed}.log')
