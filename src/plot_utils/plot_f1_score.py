import matplotlib.pyplot as plt
import numpy as np

# Situations
# situations = ['1x', '2x', '4x', '6x', '8x', '10x']
situations = ['Weighted CE Loss', 'Focal Loss (gamma = 2)', 'Focal Loss (gamma = 3)', 'Focal Loss (gamma = 4)']

# Metrics labels
metrics_labels = ['Accuracy', 'Macro F1', 'Weighted F1']

# Data for Situation 1
# situation1_data = [0.521, 0.089, 0.684]
# situation2_data = [0.609, 0.099, 0.756]
# situation3_data = [0.707, 0.108, 0.827]
# situation4_data = [0.791, 0.115, 0.882]
# situation5_data = [0.822, 0.118, 0.901]
# situation6_data = [0.859, 0.121, 0.923]
# situation1_data = [0.49, 0.09, 0.66]
# situation2_data = [0.50, 0.09, 0.67]
# situation3_data = [0.48, 0.08, 0.64]
# situation4_data = [0.51, 0.09, 0.68]
situation1_data = [0.76, 0.76, 0.77]
situation2_data = [0.76, 0.76, 0.77]
situation3_data = [0.73, 0.75, 0.74]
situation4_data = [0.73, 0.75, 0.74]


# Combine all data
# all_data = np.vstack([situation1_data, situation2_data, situation3_data, situation4_data, situation5_data, situation6_data])
all_data = np.vstack([situation1_data, situation2_data, situation3_data, situation4_data])

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Setting the positions and width for the bars
pos = np.arange(len(situations))
bar_width = 0.25

for i, metric in enumerate(metrics_labels):
    bars = ax.bar(pos + i * bar_width, all_data[:, i], bar_width, label=metric)

    # Adding data labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), ha='center', va='bottom')

# Adding labels and title
ax.set_xticks(pos + bar_width)
ax.set_xticklabels(situations)
ax.set_ylabel('Score')
ax.set_ylim(0, 1)
ax.set_title('Metrics Across Different Situations')
ax.legend()

plt.tight_layout()
plt.savefig('plot.png', bbox_inches = 'tight')

