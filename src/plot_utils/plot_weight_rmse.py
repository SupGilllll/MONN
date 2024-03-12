import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data provided by the user
x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
y1 = [6.823, 1.669, 1.652, 1.650, 1.658, 1.652, 1.641, 1.649, 1.676, 1.632, 1.669]
y2 = [6.841, 1.676, 1.684, 1.657, 1.650, 1.673, 1.640, 1.651, 1.657, 1.650, 1.673]
y3 = [6.877, 1.680, 1.673, 1.655, 1.681, 1.674, 1.654, 1.673, 1.661, 1.646, 1.703]

# Creating a DataFrame
df = pd.DataFrame({'x': x*3, 
                   'y': y1 + y2 + y3, 
                   'Threshold': ['0.3']*len(x) + ['0.4']*len(x) + ['0.5']*len(x)})

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='x', y='y', hue='Threshold', style='Threshold', markers=True, dashes=False, palette='mako')
plt.title('Performance of binding affinity prediction')
plt.xlabel('$\\alpha$')
plt.ylabel('RMSE')
# plt.ylim(1, 2)
plt.savefig('plot.png', bbox_inches = 'tight')
