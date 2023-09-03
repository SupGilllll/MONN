import os
os.chdir('/data/zhao/MONN/src')

thresholds = [0.3, 0.4, 0.5]
out_path = "../results/0807/transformer_graph"

for threshold in thresholds:
    os.system(f'python transformer_train_graph.py --clu_thre {threshold} --lr 0.00085 --d_model 224 --nhead 1 \
               --activation elu --optimizer RAdam --scheduler StepLR_10 &> {out_path}/KIKD_new_protein_{threshold}.log')