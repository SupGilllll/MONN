import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# plt.rcParams.update({'font.size': 12}) 
# Your data
x = ['0.3', '0.4', '0.5']
# y1 = [1.669, 1.653, 1.703]
# y2 = [1.649, 1.651, 1.673]
y1 = [1.656, 1.669, 1.673]
y2 = [1.654, 1.678, 1.674]

# Creating a DataFrame for Seaborn
df = pd.DataFrame({'x': x * 2, 'y': y1 + y2, 'Models': ['Single-task']*len(y1) + ['Multi-task']*len(y2)})

# Create the bar plot with the custom palette
sns.barplot(x='x', y='y', hue='Models', data=df, palette='mako')

# Customizing the plot
# plt.title('Performance comparison of binding affinity prediction', pad = 20)
plt.title('Performance comparison of binding affinity prediction')
plt.xlabel('Threshold')
plt.ylabel('RMSE')
plt.ylim(1.4, 1.8)

# plt.savefig('plot.png', bbox_inches = 'tight')
plt.tight_layout()
plt.savefig('plot.png')
