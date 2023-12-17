import os
os.chdir('/data/zhao/MONN/src')

times = [2, 4, 6, 8, 10]
out_path = "../results/1218/label_weight"


for multiple in times:
    weight = multiple * 1.4e-4
    os.system(f'python transformer_train_multiclass.py --label_weight {weight} &> {out_path}/ALL_new_new_0.4_multiple{multiple}.log')
