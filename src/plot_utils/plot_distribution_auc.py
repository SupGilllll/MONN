import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

auc_scores = np.load('../../results/240116/transformer/ALL_new_new_thre0.5_auc_list.npy')

# Create the distribution plot
sns.displot(auc_scores, color='darkred')

# Optional: Add titles and labels
plt.title('Distribution of AUC Scores (Threshold = 0.5)')
plt.xlabel('AUC Score')
plt.ylabel('Count')
# plt.legend()

# Display the plot
plt.tight_layout()
plt.savefig('plot.png')