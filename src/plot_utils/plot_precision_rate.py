import matplotlib.pyplot as plt
import numpy as np

# X-axis labels
x_labels = ['Hydrogen Bonds', 'Water Bridges', 'Hydrophobic Interactions',  
            'pi-Stacking', 'pi-Cation Interactions', 'Salt Bridges', 'Halogen Bonds']

# Data
# group1_data = [0.52, 0.59, 0.33, 0.71, 0.91, 0.89, 0.99, 1.00]
# group2_data = [0.61, 0.48, 0.40, 0.63, 0.90, 0.85, 0.98, 1.00]
# group3_data = [0.71, 0.46, 0.29, 0.48, 0.91, 0.79, 0.98, 0.98]
# group4_data = [0.79, 0.33, 0.24, 0.30, 0.83, 0.76, 0.98, 0.90]
# group5_data = [0.83, 0.25, 0.23, 0.24, 0.87, 0.61, 0.97, 0.87]
# group6_data = [0.86, 0.21, 0.21, 0.11, 0.77, 0.58, 0.97, 0.86]

group1_data = [0.67, 0.34, 0.88, 0.80, 0.65, 0.94, 1.00]
group2_data = [0.69, 0.32, 0.88, 0.82, 0.67, 0.94, 1.00]
group3_data = [0.58, 0.34, 0.89, 0.81, 0.67, 0.94, 1.00]
group4_data = [0.57, 0.37, 0.88, 0.81, 0.67, 0.94, 1.00]


# Plotting
plt.figure(figsize=(12, 6))
# plt.plot(x_labels, group1_data, label='1x', marker='o')
# plt.plot(x_labels, group2_data, label='2x', marker='o')
# plt.plot(x_labels, group3_data, label='4x', marker='o')
# plt.plot(x_labels, group4_data, label='6x', marker='o')
# plt.plot(x_labels, group5_data, label='8x', marker='o')
# plt.plot(x_labels, group6_data, label='10x', marker='o')

plt.plot(x_labels, group1_data, label='Weighted CE Loss', marker='o')
plt.plot(x_labels, group2_data, label='Focal Loss (gamma = 2)', marker='o')
plt.plot(x_labels, group3_data, label='Focal Loss (gamma = 3)', marker='o')
plt.plot(x_labels, group4_data, label='Focal Loss (gamma = 4)', marker='o')

plt.xticks(rotation=45, ha='right')
plt.ylabel('Recall Rate')
plt.title('F1 score for Different Interaction Types')
plt.legend()
plt.tight_layout()
plt.savefig('plot.png', bbox_inches = 'tight')

