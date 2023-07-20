import matplotlib.pyplot as plt
import numpy as np

def extract_numbers_from_log_file(file_path):
    total_loss_list = []
    affinity_loss_list = []
    pairwise_loss_list = []
    rmse_list = []
    auc_list = []

    with open(file_path, 'r') as file:
        for line in file.readlines()[:441]:
            if "total loss" in line:
                total_loss = float(line.split("total loss")[1].split()[0])
                total_loss_list.append(total_loss)
            
            if "affinity loss" in line:
                affinity_loss = float(line.split("affinity loss")[1].split()[0])
                affinity_loss_list.append(affinity_loss)
            
            if "pairwise loss" in line:
                pairwise_loss = float(line.split("pairwise loss")[1].split()[0])
                pairwise_loss_list.append(pairwise_loss * 0.1)
            
            if "RMSE" in line and "valid" in line:
                rmse = float(line.split("RMSE")[1].split()[0])
                rmse_list.append(rmse * 50)
            
            if "avg pairwise AUC" in line and "valid" in line:
                auc = float(line.split("avg pairwise AUC")[1].split()[0])
                auc_list.append(auc * 100)

    return total_loss_list, affinity_loss_list, pairwise_loss_list, rmse_list, auc_list

def plot_data(total_loss, affinity_loss, pairwise_loss, rmse, auc):
    epochs = range(1, len(total_loss) + 1)
    fig, ax = plt.subplots()

    # Plotting total loss
    ax.plot(epochs, total_loss, label='Total Loss')
    # Plotting affinity loss
    ax.plot(epochs, affinity_loss, label='Affinity Loss')
    # Plotting pairwise loss
    ax.plot(epochs, pairwise_loss, label='Pairwise Loss')
    # Plotting RMSE
    ax.plot(epochs, rmse, label='RMSE')
    # Plotting AUC
    ax.plot(epochs, auc, label='AUC')
    
    ax.set_ylim(0, 120)
    plt.xlabel('Epoch')
    plt.ylabel('Trend')
    plt.title('Loss Curve')
    plt.legend()
    plt.xticks(np.arange(min(epochs), max(epochs) + 1, 5))
    plt.savefig('plot.png')


log_file_path = '../../results/0612/blosum62/KIKD_new_protein_0.3.log'
total_loss, affinity_loss, pairwise_loss, rmse, auc = extract_numbers_from_log_file(log_file_path)

plot_data(total_loss, affinity_loss, pairwise_loss, rmse, auc)

