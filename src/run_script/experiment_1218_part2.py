import os
os.chdir('/data/zhao/MONN/src')

epochs = [30]
out_path = "../results/1218/new_weight"

for epoch in epochs:
    os.system(f'python transformer_train_multiclass_gpu0.py --epochs {epoch} &> {out_path}/new_weight3_epochs{epoch}.log')
