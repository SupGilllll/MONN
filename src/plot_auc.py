import matplotlib.pyplot as plt
import numpy as np

# AUC values and corresponding models
auc_values = [0.81310512, 0.7988809, 0.78757983, 0.77430532]
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
ax.set_ylim(0.75, 0.82)
ax.set_xlabel('Threshold')
ax.set_ylabel('AUC')
plt.title('AUC under new-new setting')

# Display the plot
plt.savefig('plot.png')