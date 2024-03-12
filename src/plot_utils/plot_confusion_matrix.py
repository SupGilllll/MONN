import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'font.size': 12}) 
confusion_matrix_data = np.array([[109453403, 18359223, 20616268, 34131899, 10050311, 13329261, 3090639, 4505734],
 [1344, 23108, 25856, 0, 386, 864, 1494, 0],
 [671, 5704, 9505, 0, 261, 474, 1113, 0],
 [5484, 0, 0, 48816, 10388, 2037, 2, 0],
 [169, 128, 99, 580, 33786, 2987, 2, 0],
 [124, 20, 238, 52, 492, 7114, 42, 0],
 [70, 104, 143, 0, 0, 6, 24559, 3],
 [6, 0, 0, 0, 0, 0, 0, 1223]])

labels = ['Non-interaction', 'Hydrogen Bonds', 'Water Bridges', 'Hydrophobic Interactions', 
          'pi-Stacking', 'pi-Cation Interactions', 'Salt Bridges', 'Halogen Bonds']

plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (threshold = 0.4)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')  # Rotate the x labels for better readability
plt.yticks(rotation=45, ha='right')  # Rotate the y labels for better readability
plt.savefig('plot.png', bbox_inches = 'tight')