import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'font.size': 10}) 
# Sample data
x = ['0.3', '0.4', '0.5']
y1 = [0.839, 0.822, 0.820]
y2 = [0.846, 0.842, 0.842]
# y3 = [0.803, 0.808, 0.806]
# y1 = [0.810, 0.793, 0.792]
# y2 = [0.835, 0.833, 0.832]

# Creating a DataFrame for Seaborn
df = pd.DataFrame({'x': x * 2, 'y': y1 + y2, 'Models': ['MONN']*len(y1) + ['Proposed model']*len(y2)})
# df = pd.DataFrame({'x': x * 3, 'y': y1 + y2 + y3, 'Models': ['MONN']*len(y1) + ['Proposed model (binary)']*len(y2) + ['Proposed model (multi-class)']*len(y3)})

# Create the bar plot with the custom palette
sns.barplot(x='x', y='y', hue='Models', data=df, palette='rocket')

# Customizing the plot
# plt.title('Performance comparison of non-covalent interaction prediction (pocket region)', pad=20, fontsize=10)
plt.title('Performance comparison of non-covalent interaction prediction', pad=20, fontsize=10)
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.ylim(0.75, 0.9)

plt.savefig('plot.png', bbox_inches = 'tight')
