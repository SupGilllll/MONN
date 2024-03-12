import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12}) 
# Sample data
x = ['0.3', '0.4', '0.5']
y1 = [1.574, 1.613, 1.594]
y2 = [1.649, 1.651, 1.673]
# y3 = [1.658, 1.678, 1.696]

# Creating a DataFrame for Seaborn
df = pd.DataFrame({'x': x * 2, 'y': y1 + y2, 'Models': ['MONN']*len(y1) + ['Proposed model']*len(y2)})
# df = pd.DataFrame({'x': x * 3, 'y': y1 + y2 + y3, 'Models': ['MONN']*len(y1) + ['Proposed model (binary)']*len(y2) + ['Proposed model (multi-class)']*len(y3)})

# Create the bar plot with the custom palette
sns.barplot(x='x', y='y', hue='Models', data=df, palette='mako')

# Customizing the plot
# plt.title('Performance comparison of binding affinity prediction')
plt.title('Performance comparison of binding affinity prediction', pad=20)
plt.xlabel('Threshold')
plt.ylabel('RMSE')
plt.ylim(1.4, 1.8)

# Show the plot
plt.tight_layout()
plt.savefig('plot.png')
