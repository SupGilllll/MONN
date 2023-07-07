import matplotlib.pyplot as plt

# Sample data
x_values = ['0.3', '0.4', '0.5', '0.6']
correlation_values = [0.52729021, 0.49591197, 0.46304611, 0.48075808]
rmse_values = [1.7377026, 1.7797419, 1.81391531, 1.72286034]

# Create a figure and axis
fig, ax = plt.subplots()

# Create scatterplot for correlation
ax.plot(x_values, correlation_values, marker='o', linestyle='-', color='b', label='Pearson Correlation')

# Create scatterplot for RMSE
ax.plot(x_values, rmse_values, marker='o', linestyle='-', color='r', label='RMSE')

# Set labels and title
ax.set_xlabel('Threshold')
ax.set_ylabel('Value')
plt.title('Pearson Correlation and RMSE under new-new setting')
# ax.set_xticks(x_values)
# Add legend
plt.legend()

# Save the plot as an image file (e.g., PNG, JPEG, SVG, PDF)
plt.savefig('plot.png')

