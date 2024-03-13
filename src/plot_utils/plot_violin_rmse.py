import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14}) 
# Generating mock data
label1 = np.load('../../results/240116/transformer/ALL_new_new_thre0.3_aff_label.npy')
pred1 = np.load('../../results/240116/transformer/ALL_new_new_thre0.3_aff_pred.npy')
label2 = np.load('../../results/240116/transformer/ALL_new_new_thre0.4_aff_label.npy')
pred2 = np.load('../../results/240116/transformer/ALL_new_new_thre0.4_aff_pred.npy')
label3 = np.load('../../results/240116/transformer/ALL_new_new_thre0.5_aff_label.npy')
pred3 = np.load('../../results/240116/transformer/ALL_new_new_thre0.5_aff_pred.npy')
errors_threshold_1 = np.abs(label1 - pred1)
errors_threshold_2 = np.abs(label2 - pred2)
errors_threshold_3 = np.abs(label3 - pred3)

data_length = len(errors_threshold_1)
# Creating a DataFrame
data = {
    'Error Value': np.concatenate([errors_threshold_1, errors_threshold_2, errors_threshold_3]),
    'Threshold': ['0.3']*data_length + ['0.4']*data_length + ['0.5']*data_length
}
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
sns.violinplot(x='Threshold', y='Error Value', data=df, palette='mako', hue = 'Threshold')
plt.title('Distribution of Errors', pad = 20)
plt.xlabel('Threshold')
plt.ylabel('Error')
plt.savefig('plot.png', bbox_inches = 'tight')