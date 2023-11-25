import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data
x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# y1 = [0.816, 0.841, 0.843, 0.838, 0.839, 0.842, 0.842, 0.844, 0.841, 0.838, 0.511] # 0.7 0.6 0.5
# y2 = [0.808, 0.842, 0.841, 0.839, 0.839, 0.836, 0.833, 0.830, 0.832, 0.821, 0.470] # 0.1 0.2 
# y3 = [0.815, 0.830, 0.834, 0.834, 0.837, 0.834, 0.835, 0.834, 0.834, 0.833, 0.511] # 0.4 0.6
y1 = [6.595, 1.851, 1.817, 1.823, 1.822, 1.812, 1.805, 1.786, 1.799, 1.786, 1.821] # 0.9 0.7 0.8
y2 = [6.587, 1.854, 1.832, 1.830, 1.862, 1.843, 1.836, 1.827, 1.838, 1.820, 1.868] # 0.9 0.7 0.8
y3 = [6.661, 1.875, 1.854, 1.869, 1.853, 1.856, 1.845, 1.845, 1.870, 1.857, 1.883] # 0.7 0.6 0.4

# Convert data to DataFrame
df = pd.DataFrame({
    'X': x,
    '0.3': y1,
    '0.4': y2,
    '0.5': y3
})

# Melting the DataFrame to long format for seaborn
df_long = pd.melt(df, 'X', var_name='Thresholds', value_name='Value')

# Plot
plt.figure(figsize=(12, 6))

sns.lineplot(data=df_long, x='X', y='Value', hue='Thresholds')
plt.title('Performance with different loss weight')
plt.xlabel('p (loss weight)')
plt.ylabel('RMSE')
plt.ylim(1.6, 2.6)
plt.savefig('plot.png', bbox_inches = 'tight')
