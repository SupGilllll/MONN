import os
os.chdir('/data/zhao/MONN/src')

thresholds = [0.3, 0.4, 0.5]
out_path = "../results/0807/transformer_novel"

# trials = [33, 37, 40, 50, 120]
for threshold in thresholds:
    os.system(f'python transformer_train_novel.py --clu_thre {threshold} --lr 0.00025 --d_model 192 --nhead 4 \
                --activation gelu --optimizer Adam --scheduler StepLR_10 &> {out_path}/KIKD_new_protein_{threshold}_trial_1_33.log')
    os.system(f'python transformer_train_novel.py --clu_thre {threshold} --lr 0.00025 --d_model 192 --nhead 4 \
                --activation leaky_relu --optimizer Adam --scheduler none &> {out_path}/KIKD_new_protein_{threshold}_trial_1_37.log')
    os.system(f'python transformer_train_novel.py --clu_thre {threshold} --lr 0.0006 --d_model 192 --nhead 4 \
                --activation leaky_relu --optimizer Adam --scheduler none &> {out_path}/KIKD_new_protein_{threshold}_trial_1_40.log')
    os.system(f'python transformer_train_novel.py --clu_thre {threshold} --lr 0.0006 --d_model 192 --nhead 4 \
                --activation leaky_relu --optimizer Adam --scheduler ReduceLROnPlateau &> {out_path}/KIKD_new_protein_{threshold}_trial_1_50.log')
    os.system(f'python transformer_train_novel.py --clu_thre {threshold} --lr 0.00025 --d_model 224 --nhead 4 \
                --activation leaky_relu --optimizer Adam --scheduler none &> {out_path}/KIKD_new_protein_{threshold}_trial_1_120.log')

            
# trials = [86, 124, 64, 82, 87, 99, 115, 119, 129, 138]
for threshold in thresholds:
    os.system(f'python transformer_train_novel.py --clu_thre {threshold} --lr 0.00065 --d_model 128 --nhead 8 \
                --activation leaky_relu --optimizer Adam --scheduler none &> {out_path}/KIKD_new_protein_{threshold}_trial_2_86.log')
    os.system(f'python transformer_train_novel.py --clu_thre {threshold} --lr 0.0008 --d_model 128 --nhead 8 \
                --activation leaky_relu --optimizer Adam --scheduler ReduceLROnPlateau &> {out_path}/KIKD_new_protein_{threshold}_trial_2_124.log')
    os.system(f'python transformer_train_novel.py --clu_thre {threshold} --lr 0.00045 --d_model 128 --nhead 8 \
                --activation leaky_relu --optimizer Adam --scheduler ReduceLROnPlateau &> {out_path}/KIKD_new_protein_{threshold}_trial_2_64.log')
    os.system(f'python transformer_train_novel.py --clu_thre {threshold} --lr 0.00065 --d_model 128 --nhead 8 \
                --activation leaky_relu --optimizer Adam --scheduler ReduceLROnPlateau &> {out_path}/KIKD_new_protein_{threshold}_trial_2_82.log')
    os.system(f'python transformer_train_novel.py --clu_thre {threshold} --lr 0.0003 --d_model 128 --nhead 8 \
                --activation leaky_relu --optimizer Adam --scheduler none &> {out_path}/KIKD_new_protein_{threshold}_trial_2_99.log')
    os.system(f'python transformer_train_novel.py --clu_thre {threshold} --lr 0.0005 --d_model 128 --nhead 8 \
                --activation leaky_relu --optimizer Adam --scheduler StepLR_1 &> {out_path}/KIKD_new_protein_{threshold}_trial_2_138.log')