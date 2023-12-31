import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Data
x = ['0.3', '0.4', '0.5']
# y1 = [0.834, 0.828, 0.820]
# y2 = [0.844, 0.842, 0.841]
# y3 = [0.804, 0.808, 0.800]
# y1 = [0.844, 0.839, 0.828]
# y2 = [0.861, 0.862, 0.861]
# y3 = [0.829, 0.832, 0.823]
y1 = [0.825, 0.821, 0.806]
y2 = [0.860, 0.861, 0.858]
y3 = [0.826, 0.831, 0.817]
# x = ['0.4']
# y1 = [0.844]
# y2 = [0.829]
# y3 = [0.851]

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
ax.bar(index - bar_width, y1, bar_width, label='Baseline', align='center', color=colors1)
ax.bar(index, y2, bar_width, label='Proposed Model (binary classification)', align='center', color=colors2)
ax.bar(index + bar_width, y3, bar_width, label='Proposed Model (multi-class classification)', align='center', color=colors3)

# Set labels, title, and legend
ax.set_xlabel('Threshold')
ax.set_ylabel('AUC')
ax.set_title('Type3 AUC under both-new setting (ALL)')
# ax.set_title('AUC on pocket area')
ax.set_xticks(index)
ax.set_xticklabels(x_numeric)
ax.set_ylim(0.78, 0.90)
ax.legend(fontsize=12)

# Show the plot
plt.tight_layout()
plt.savefig('plot.png')
