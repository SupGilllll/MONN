import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = ['0.3', '0.4', '0.5']
y1 = [0.81321491, 0.80459854, 0.79266345]
y2 = [0.91108614, 0.90519285, 0.90158103]

# Set up the figure and axis
fig, ax = plt.subplots()

# Define the width of each bar
bar_width = 0.35

# Set the positions of the x-axis ticks
x_pos = np.arange(len(x))

# Create the bars for the first y-axis data
ax.bar(x_pos, y1, width=bar_width, color='blue', label='Original Model')

# Shift the x-axis positions for the second set of bars
x_pos_shifted = x_pos + bar_width

# Create the bars for the second y-axis data
ax.bar(x_pos_shifted, y2, width=bar_width, color='orange', label='Modified')

# Set the x-axis tick positions and labels
ax.set_xticks(x_pos + bar_width / 2)
ax.set_xticklabels(x)
ax.set_ylim(0.75, 0.95)

# Set the y-axis label
ax.set_xlabel('Threshold')
ax.set_ylabel('AUC')

# Add a legend
ax.legend()

# Show the plot
plt.savefig('plot.png')