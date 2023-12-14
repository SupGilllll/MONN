import matplotlib.pyplot as plt
import numpy as np

def extract_numbers_from_log_file(file_path, fold = 1):
    rmse_list = []
    auc_list = []
    training_loss_list = []
    validation_loss_list = []
    cnt = 0
    
    with open(file_path, 'r') as file:
        for line in file.readlines():
            if "train num:" in line:
                cnt += 1

            if cnt != fold:
                continue

            if "training loss total loss" in line:
                training_loss = float(line.split("training loss total loss")[1].split()[0])
                training_loss_list.append(training_loss)

            if "valiation loss total loss" in line:
                validation_loss = float(line.split("valiation loss total loss")[1].split()[0])
                validation_loss_list.append(validation_loss)

            if "validation loss total loss" in line:
                validation_loss = float(line.split("validation loss total loss")[1].split()[0])
                validation_loss_list.append(validation_loss)
            
            # if "RMSE" in line and "valid" in line:
            #     rmse = float(line.split("RMSE")[1].split()[0])
            #     rmse_list.append(rmse * 5 + 70)
            
            # if "avg pairwise AUC" in line and "valid" in line:
            #     auc = float(line.split("avg pairwise AUC")[1].split()[0])
            #     auc_list.append(auc * 5 + 70)

    # return total_loss_list, affinity_loss_list, pairwise_loss_list, rmse_list, auc_list
    return training_loss_list, validation_loss_list, rmse_list, auc_list

def plot_data(training_loss, validation_loss, rmse, auc):
    epochs = range(0, len(training_loss[:30]))
    fig, ax = plt.subplots()

    # Plotting total loss
    ax.plot(epochs, training_loss[:30], label='Training_loss')
    # Plotting affinity loss
    ax.plot(epochs, validation_loss[:30], label='Validation_loss')
    # Plotting RMSE
    # ax.plot(epochs, rmse[:30], label='RMSE')
    # # Plotting AUC
    # ax.plot(epochs, auc[:30], label='AUC')
    
    ax.set_ylim(0, 20)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend(fontsize=14)
    plt.xticks(np.arange(min(epochs), max(epochs) + 1, 5))
    plt.savefig('plot.png')


# log_file_path = '../../results/0724/transformer_base_seed42/KIKD_new_protein_0.3.log'
log_file_path = '/data/zhao/MONN/results/1204/multi-class/ALL_new_new_0.4.log'
# total_loss, affinity_loss, pairwise_loss, rmse, auc = extract_numbers_from_log_file(log_file_path)
training_loss, validation_loss, rmse, auc = extract_numbers_from_log_file(log_file_path, 2)

# print(training_loss)
# print(validation_loss)
# print(rmse)
# print(auc)
# plot_data(total_loss, affinity_loss, pairwise_loss, rmse, auc)
plot_data(training_loss, validation_loss, rmse, auc)

