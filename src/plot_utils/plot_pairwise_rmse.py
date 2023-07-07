import matplotlib.pyplot as plt

# Sample data
x_values = ['0.3', '0.4', '0.5']
correlation_values1 = [0.50981112, 0.49461565, 0.46512096]
rmse_values1 = [1.75455851, 1.77652398, 1.81545774]
correlation_values2 = [0.5143357, 0.49600565, 0.4739919]
rmse_values2 = [1.75722742, 1.79249492, 1.81875831]

# Create a figure and axis
fig, ax = plt.subplots()

# Create scatterplot for correlation
ax.plot(x_values, correlation_values1, marker='o', linestyle='-', color='green', label='Pearson Correlation (original)')

# Create scatterplot for RMSE
ax.plot(x_values, rmse_values1, marker='o', linestyle='-', color='green', label='RMSE (original)')

# Create scatterplot for correlation
ax.plot(x_values, correlation_values2, marker='o', linestyle='-', color='purple', label='Pearson Correlation (modified)')

# Create scatterplot for RMSE
ax.plot(x_values, rmse_values2, marker='o', linestyle='-', color='purple', label='RMSE (modified)')

# Set labels and title
ax.set_xlabel('Threshold')
ax.set_ylabel('Value')
plt.title('Pearson Correlation and RMSE under new-new setting (IC50)')
# ax.set_xticks(x_values)
# Add legend
plt.legend()

# Save the plot as an image file (e.g., PNG, JPEG, SVG, PDF)
plt.savefig('plot.png')

