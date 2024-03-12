import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# plt.rcParams.update({'font.size': 12}) 
# Your data
x = ['0.3', '0.4', '0.5']
# y1 = [0.808, 0.807, 0.806]
# y2 = [0.846, 0.842, 0.842]
y1 = [0.808, 0.807, 0.805]
y2 = [0.844, 0.841, 0.841]

# Creating a DataFrame for Seaborn
df = pd.DataFrame({'x': x * 2, 'y': y1 + y2, 'Models': ['Single-task']*len(y1) + ['Multi-task']*len(y2)})

# Create the bar plot with the custom palette
sns.barplot(x='x', y='y', hue='Models', data=df, palette='rocket')

# Customizing the plot
# plt.title('Performance comparison of non-covalent interaction prediction', pad = 20)
plt.title('Performance comparison of non-covalent interaction prediction')
plt.xlabel('Threshold')
plt.ylabel('AUC')
plt.ylim(0.75, 0.9)

# plt.savefig('plot.png', bbox_inches = 'tight')
plt.tight_layout()
plt.savefig('plot.png')