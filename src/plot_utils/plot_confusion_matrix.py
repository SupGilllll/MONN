import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

confusion_matrix_data = np.array([
    [110841110, 24748772, 12769120, 32261223, 10190583, 14498493, 3729236, 4498201],
    [1757, 31067, 16489, 0, 454, 1046, 2007, 0],
    [561, 9235, 6039, 0, 274, 534, 1395, 0],
    [6358, 0, 0, 47279, 11158, 2184, 0, 0],
    [177, 31, 71, 232, 34150, 2785, 6, 0],
    [50, 18, 90, 12, 676, 7207, 85, 0],
    [64, 94, 61, 0, 2, 8, 24564, 5],
    [1, 0, 0, 0, 0, 0, 0, 1228]
])

labels = ['Non-interaction', 'Hydrogen Bonds', 'Water Bridges', 'Hydrophobic Interactions', 
          'pi-Stacking', 'pi-Cation Interactions', 'Salt Bridges', 'Halogen Bonds']

plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')  # Rotate the x labels for better readability
plt.yticks(rotation=45, ha='right')  # Rotate the y labels for better readability
plt.savefig('plot.png', bbox_inches = 'tight')