import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data provided by the user
x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
y1 = [0.839, 0.844, 0.845, 0.845, 0.846, 0.844, 0.846, 0.846, 0.845, 0.844, 0.446]
y2 = [0.834, 0.842, 0.838, 0.841, 0.841, 0.838, 0.837, 0.842, 0.842, 0.840, 0.436]
y3 = [0.836, 0.840, 0.842, 0.838, 0.841, 0.837, 0.840, 0.842, 0.841, 0.841, 0.413]

# Creating a DataFrame
df = pd.DataFrame({'x': x*3, 
                   'y': y1 + y2 + y3, 
                   'Threshold': ['0.3']*len(x) + ['0.4']*len(x) + ['0.5']*len(x)})

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='x', y='y', hue='Threshold', style='Threshold', markers=True, dashes=False, palette='rocket')
plt.title('Performance of non-covalent interaction prediction')
plt.xlabel('$\\alpha$')
plt.ylabel('AUC')
plt.ylim(0.4, 0.9)
plt.savefig('plot.png', bbox_inches = 'tight')
