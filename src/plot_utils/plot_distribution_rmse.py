import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

label = np.load('../../results/240116/transformer/ALL_new_new_thre0.5_aff_label.npy')
pred = np.load('../../results/240116/transformer/ALL_new_new_thre0.5_aff_pred.npy')
error = np.abs(label - pred)

# Create the distribution plot
sns.displot(error, color='darkblue')

# Optional: Add titles and labels
plt.title('Distribution of Error (Threshold = 0.5)')
plt.xlabel('Error')
plt.ylabel('Count')
# plt.legend()

# Display the plot
plt.tight_layout()
plt.savefig('plot.png')