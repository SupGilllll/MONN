file_name = '/data/zhao/MONN/results/240116/transformer_trial2/loss_weight/ALL_new_new_0.3_0.1_epochs30.log'
flag = False
# weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# for weight in weights:
#     rmse, avg_auc, fold_auc = 0, 0, 0
#     with open(f'/data/zhao/MONN/results/240116/transformer_trial2/loss_weight/ALL_new_new_0.5_{weight}_epochs30.log', 'r') as f:
#         for line in f.readlines():
#             if 'validation loss' in line and line.startswith('epoch: 29'):
#                 flag = True
#                 continue
#             if line.startswith('test') and flag:
#                 flag = False
#                 str = line.strip().split()
#                 rmse += float(str[3])
#                 avg_auc += float(str[11])
#                 fold_auc += float(str[14])

#     rmse = rmse / 9
#     avg_auc = avg_auc / 9
#     fold_auc = fold_auc / 9

#     print(rmse, avg_auc, fold_auc)

# thresholds = [0.3, 0.4, 0.5]

# for threshold in thresholds:
#     rmse, avg_auc, fold_auc = 0, 0, 0
#     with open(f'/data/zhao/MONN/results/240116/transformer_trial2/binary_binding/ALL_new_new_{threshold}_epochs30.log', 'r') as f:
#         for line in f.readlines():
#             if 'validation loss' in line and line.startswith('epoch: 29'):
#                 flag = True
#                 continue
#             if line.startswith('test') and flag:
#                 flag = False
#                 str = line.strip().split()
#                 rmse += float(str[3])
#                 # avg_auc += float(str[11])
#                 # fold_auc += float(str[14])

#     rmse = rmse / 9
#     avg_auc = avg_auc / 9
#     fold_auc = fold_auc / 9

#     print(rmse, avg_auc, fold_auc)

thresholds = [0.3, 0.4, 0.5]

for threshold in thresholds:
    rmse, avg_auc, fold_auc = 0, 0, 0
    with open(f'/data/zhao/MONN/results/240116/transformer/multi_class/ALL_new_new_{threshold}.log', 'r') as f:
        for line in f.readlines():
            if 'validation loss' in line and line.startswith('epoch: 15'):
                flag = True
                continue
            if line.startswith('test') and flag:
                flag = False
                str = line.strip().split()
                rmse += float(str[3])
                avg_auc += float(str[11])
                fold_auc += float(str[14])

    rmse = rmse / 9
    avg_auc = avg_auc / 9
    fold_auc = fold_auc / 9

    print(rmse, avg_auc, fold_auc)
