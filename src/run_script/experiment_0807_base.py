import os
os.chdir('/data/zhao/MONN/src')

thresholds = [0.3, 0.4, 0.5]
# out_path = "../results/0807/transformer_base"

# trials = [10, 23, 44, 66, 74, 85, 91, 108]
# for threshold in thresholds:
    # os.system(f'python transformer_train.py --clu_thre {threshold} --lr 0.0003 --num_layers 2 --d_model 160 --nhead 8 \
    #             --activation gelu --optimizer Adam --scheduler StepLR_1 &> {out_path}/KIKD_new_protein_{threshold}_trial_1_10.log')
    # os.system(f'python transformer_train.py --clu_thre {threshold} --lr 0.00045 --num_layers 2 --d_model 128 --nhead 8 \
    #             --activation gelu --optimizer Adam --scheduler StepLR_1 &> {out_path}/KIKD_new_protein_{threshold}_trial_1_23.log')
    # os.system(f'python transformer_train.py --clu_thre {threshold} --lr 0.0002 --num_layers 2 --d_model 160 --nhead 8 \
    #             --activation gelu --optimizer RAdam --scheduler StepLR_1 &> {out_path}/KIKD_new_protein_{threshold}_trial_1_44.log')
    # os.system(f'python transformer_train.py --clu_thre {threshold} --lr 0.00025 --num_layers 2 --d_model 160 --nhead 8 \
    #             --activation gelu --optimizer Adam --scheduler none &> {out_path}/KIKD_new_protein_{threshold}_trial_1_66.log')
    # os.system(f'python transformer_train.py --clu_thre {threshold} --lr 0.0005 --num_layers 2 --d_model 160 --nhead 8 \
    #             --activation gelu --optimizer Adam --scheduler StepLR_1 &> {out_path}/KIKD_new_protein_{threshold}_trial_1_74.log')
    # os.system(f'python transformer_train.py --clu_thre {threshold} --lr 0.0003 --num_layers 2 --d_model 160 --nhead 8 \
    #             --activation gelu --optimizer Adam --scheduler none &> {out_path}/KIKD_new_protein_{threshold}_trial_1_85.log')
    # os.system(f'python transformer_train.py --clu_thre {threshold} --lr 0.00055 --num_layers 2 --d_model 160 --nhead 8 \
    #             --activation gelu --optimizer Adam --scheduler StepLR_1 &> {out_path}/KIKD_new_protein_{threshold}_trial_1_91.log')
    # os.system(f'python transformer_train.py --clu_thre {threshold} --lr 0.0004 --num_layers 2 --d_model 160 --nhead 8 \
    #             --activation gelu --optimizer Adam --scheduler StepLR_1 &> {out_path}/KIKD_new_protein_{threshold}_trial_1_108.log')
            
# trials = [17, 79, 88]
# for threshold in thresholds:
#     os.system(f'python transformer_train.py --clu_thre {threshold} --lr 0.00055 --num_layers 2 --d_model 128 --nhead 8 \
#                 --activation gelu --optimizer Adam --scheduler StepLR_1 &> {out_path}/KIKD_new_protein_{threshold}_trial_2_17.log')
#     os.system(f'python transformer_train.py --clu_thre {threshold} --lr 0.0002 --num_layers 2 --d_model 224 --nhead 8 \
#                 --activation gelu --optimizer Adam --scheduler StepLR_1 &> {out_path}/KIKD_new_protein_{threshold}_trial_2_79.log')
#     os.system(f'python transformer_train.py --clu_thre {threshold} --lr 0.00065 --num_layers 2 --d_model 192 --nhead 1 \
#                 --activation gelu --optimizer Adam --scheduler StepLR_1 &> {out_path}/KIKD_new_protein_{threshold}_trial_2_88.log')
    
# out_path = "../results/0807/transformer_base/absolute"
# for threshold in thresholds:
#     os.system(f'python transformer_train.py --clu_thre {threshold} --lr 0.0001 --epochs 20 --pos_encoding absolute &> \
#               {out_path}/KIKD_new_protein_{threshold}.log')

out_path = "../results/0807/transformer_base/trial10_1"
for threshold in thresholds:
    os.system(f'python transformer_train.py --clu_thre {threshold} --lr 0.0003 --epochs 40 --d_model 160 --nhead 8 \
               --activation gelu --optimizer Adam --scheduler StepLR_1 &> {out_path}/KIKD_new_protein_{threshold}_StepLR_1.log')
    os.system(f'python transformer_train.py --clu_thre {threshold} --lr 0.0003 --epochs 40 --d_model 160 --nhead 8 \
               --activation gelu --optimizer Adam --scheduler StepLR_10 &> {out_path}/KIKD_new_protein_{threshold}_StepLR_10.log')
    os.system(f'python transformer_train.py --clu_thre {threshold} --lr 0.0003 --epochs 40 --d_model 160 --nhead 8 \
               --activation gelu --optimizer Adam --scheduler none &> {out_path}/KIKD_new_protein_{threshold}_none.log')