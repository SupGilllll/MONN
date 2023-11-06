import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Data
x = ['0.3', '0.4', '0.5']
y1 = [0.811, 0.808, 0.791]
y2 = [0.819, 0.817, 0.804]
y3 = [0.834, 0.830, 0.819]
# x = ['0.4']
# y1 = [0.844]
# y2 = [0.829]
# y3 = [0.851]

cmap = matplotlib.colormaps.get_cmap('gist_earth')

# Convert x to numeric values for plotting
x_numeric = [float(val) for val in x]

# Create a figure and axis object
fig, ax = plt.subplots()

# Set the width of the bars
bar_width = 0.2

# Create an array of indices for the x-axis locations
index = np.arange(len(x_numeric))

# Create the bar plots
ax.plot(x, y1,'xb-', label='Baseline model')
ax.plot(x, y2,'xr-', label='Pure Transformer backbone')
ax.plot(x, y3,'xg-', label='Transformer backbone + graph network')

# Set labels, title, and legend
ax.set_xlabel('Threshold')
ax.set_ylabel('AUC')
ax.set_title('AUC under both-new setting (KIKD)')
ax.set_xticks(index)
ax.set_xticklabels(x_numeric)
ax.set_ylim(0.790, 0.850)
ax.legend(fontsize=12)

# Show the plot
plt.savefig('plot.png')
