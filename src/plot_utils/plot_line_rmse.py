import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Data
x = ['0.3', '0.4', '0.5']
y1 = [1.676, 1.706, 1.720]
y2 = [1.696, 1.705, 1.761]
y3 = [1.669, 1.728, 1.735]
# x = ['0.4']
# y1 = [1.706]
# y2 = [1.705]
# y3 = [1.728]

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
ax.plot(x, y1,'xb-', label='Baseline model')
ax.plot(x, y2,'xr-', label='Pure Transformer backbone')
ax.plot(x, y3,'xg-', label='Transformer backbone + graph network')

# Set labels, title, and legend
ax.set_xlabel('Threshold')
ax.set_ylabel('RMSE')
ax.set_title('RMSE under new-protein setting (KIKD)')
ax.set_xticks(index)
ax.set_xticklabels(x_numeric)
ax.set_ylim(1.65, 1.8)
ax.legend(fontsize=12)

# Show the plot
plt.savefig('plot.png')
