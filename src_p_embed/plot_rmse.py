import matplotlib.pyplot as plt

# Sample data
x_values = ['0.3', '0.4', '0.5', '0.6']
correlation_values = [0.54228906, 0.52974118, 0.53003064, 0.50341555]
rmse_values = [1.35876373, 1.38005795, 1.3818055, 1.39479713]

# Create a figure and axis
fig, ax = plt.subplots()

# Create scatterplot for correlation
ax.plot(x_values, correlation_values, marker='o', linestyle='-', color='b', label='Pearson Correlation')

# Create scatterplot for RMSE
ax.plot(x_values, rmse_values, marker='o', linestyle='-', color='r', label='RMSE')

# Set labels and title
ax.set_xlabel('Threshold')
ax.set_ylabel('Value')
plt.title('Pearson Correlation and RMSE under new-new setting (IC50)')
# ax.set_xticks(x_values)
# Add legend
plt.legend()

# Save the plot as an image file (e.g., PNG, JPEG, SVG, PDF)
plt.savefig('plot.png')

