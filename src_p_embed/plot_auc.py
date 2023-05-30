import matplotlib.pyplot as plt
import numpy as np

# AUC values and corresponding models
auc_values = [0.82346908, 0.81939903, 0.81721172, 0.8022592]
models = ['0.3', '0.4', '0.5', '0.6']

# Create a figure and axis
fig, ax = plt.subplots()

# Create bar chart
x_pos = np.arange(len(models))
ax.bar(x_pos, auc_values, align='center', alpha=0.8)

# Set x-axis tick positions and labels
ax.set_xticks(x_pos)
ax.set_xticklabels(models, rotation=45)

# Set labels and title
ax.set_ylim(0.80, 0.825)
ax.set_xlabel('Threshold')
ax.set_ylabel('AUC')
plt.title('AUC under new-new setting (IC50)')

# Display the plot
plt.savefig('plot.png')