import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Data
x = ['0.3', '0.4', '0.5']
y1 = [0.832, 0.824, 0.832]
y2 = [0.809, 0.809, 0.809]
# x = ['0.4']
# y1 = [0.844]
# y2 = [0.829]
# y3 = [0.851]

cmap = matplotlib.colormaps.get_cmap('gist_earth')
colors1 = [cmap(0.25)]
colors2 = [cmap(0.5)]

# Convert x to numeric values for plotting
x_numeric = [float(val) for val in x]

# Create a figure and axis object
fig, ax = plt.subplots()

# Set the width of the bars
bar_width = 0.2

# Create an array of indices for the x-axis locations
index = np.arange(len(x_numeric))

# Create the bar plots
ax.bar(index - bar_width / 2, y1, bar_width, label='Whole sequence', align='center', color=colors1)
ax.bar(index + bar_width / 2, y2, bar_width, label='Pocket area', align='center', color=colors2)

# Set labels, title, and legend
ax.set_xlabel('Threshold')
ax.set_ylabel('AUC')
ax.set_title('Graph Transformer Model')
ax.set_xticks(index)
ax.set_xticklabels(x_numeric)
ax.set_ylim(0.73, 0.85)
ax.legend(fontsize=12)

# Show the plot
plt.savefig('plot.png')