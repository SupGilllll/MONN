import matplotlib.pyplot as plt

# Assuming you have a list of 1000 data points
data = [i for i in range(1, 1001)]

# Extract the first 50 and last 50 data points
first_50 = data[:50]
last_50 = data[-50:]

# Create labels for the bars, if needed
x1_labels = [str(i) for i in range(1, 51)]  # Optional, just for labeling
x2_labels = [str(i) for i in range(101, 151)]  # Optional, just for labeling

# Create the barplot
plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
plt.bar(x1_labels, first_50, label='First 50 Data', alpha=0.7)
plt.bar(x2_labels, last_50, label='Last 50 Data', alpha=0.7, bottom=first_50)
plt.xlabel('Data Point Index')
plt.ylabel('Value')
plt.title('Barplot of First 50 and Last 50 Data Points')
plt.legend()
plt.savefig('plot.png')
