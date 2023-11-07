import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate random data for demonstration
data = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]  # Replace with your actual data

sns.kdeplot(data, fill=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Density Plot of Data')
plt.savefig('plot.png')