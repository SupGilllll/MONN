import os
os.chdir('/data/zhao/MONN/src')

gammas = [3, 4]
out_path = "../results/1218/focal"


for gamma in gammas:
    os.system(f'python transformer_train_multiclass.py --focal_gamma {gamma} &> {out_path}/ALL_new_new_0.4_gamma{gamma}.log')
