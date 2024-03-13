import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14}) 
# Generating mock data
np.random.seed(0)  # For reproducibility
errors_threshold_1 = np.load('../../results/240116/transformer/ALL_new_new_thre0.3_auc_list.npy')
errors_threshold_2 = np.load('../../results/240116/transformer/ALL_new_new_thre0.4_auc_list.npy')
errors_threshold_3 = np.load('../../results/240116/transformer/ALL_new_new_thre0.5_auc_list.npy')

data_length = len(errors_threshold_1)
# Creating a DataFrame
data = {
    'Error Value': np.concatenate([errors_threshold_1, errors_threshold_2, errors_threshold_3]),
    'Threshold': ['0.3']*data_length + ['0.4']*data_length + ['0.5']*data_length
}
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
sns.violinplot(x='Threshold', y='Error Value', data=df, palette='rocket', hue = 'Threshold')
plt.title('Distribution of AUC scores', pad = 20)
plt.xlabel('Threshold')
plt.ylabel('AUC')
plt.savefig('plot.png', bbox_inches = 'tight')