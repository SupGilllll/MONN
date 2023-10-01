import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Data
# x = ['0.3', '0.4', '0.5']
# y1 = [0.851, 0.844, 0.844]
# y2 = [0.830, 0.829, 0.830]
# y3 = [0.850, 0.851, 0.844]
x = ['0.4']
y1 = [0.844]
y2 = [0.829]
y3 = [0.851]

cmap = matplotlib.colormaps.get_cmap('gist_earth')
colors1 = [cmap(0.25)]
colors2 = [cmap(0.5)]
colors3 = [cmap(0.75)]

# Convert x to numeric values for plotting
x_numeric = [float(val) for val in x]

# Create a figure and axis object
fig, ax = plt.subplots()

# Set the width of the bars
bar_width = 0.2

# Create an array of indices for the x-axis locations
index = np.arange(len(x_numeric))

# Create the bar plots
ax.bar(index - bar_width, y1, bar_width, label='Baseline model', align='center', color=colors1)
ax.bar(index, y2, bar_width, label='Pure Transformer backbone', align='center', color=colors2)
ax.bar(index + bar_width, y3, bar_width, label='Transformer backbone + graph network', align='center', color=colors3)

# Set labels, title, and legend
ax.set_xlabel('Threshold')
ax.set_ylabel('AUC')
ax.set_title('AUC under new-protein setting (KIKD)')
ax.set_xticks(index)
ax.set_xticklabels(x_numeric)
ax.set_ylim(0.75, 0.9)
ax.legend(fontsize=12)

# Show the plot
plt.savefig('plot.png')
