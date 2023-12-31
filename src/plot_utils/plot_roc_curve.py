import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_true1 = np.load('/data/zhao/MONN/results/1227/baseline/ALL_new_new_thre0.4_label.npy')
y_scores1 = np.load('/data/zhao/MONN/results/1227/baseline/ALL_new_new_thre0.4_pred.npy')
y_true2 = np.load('/data/zhao/MONN/results/1227/binary-class/ALL_new_new_thre0.4_label.npy')
y_scores2 = np.load('/data/zhao/MONN/results/1227/binary-class/ALL_new_new_thre0.4_pred.npy')
y_true3 = np.load('/data/zhao/MONN/results/1227/multi-class/ALL_new_new_thre0.4_label.npy')
y_scores3 = np.load('/data/zhao/MONN/results/1227/multi-class/ALL_new_new_thre0.4_pred.npy')

# Function to plot ROC curve for a group
def plot_roc_curve(y_true, y_scores, label):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')

# Plotting the ROC curves
plt.figure()

plot_roc_curve(y_true1, y_scores1, 'Baseline')
plot_roc_curve(y_true2, y_scores2, 'Proposed Model (binary classification)')
plot_roc_curve(y_true3, y_scores3, 'Proposed Model (multi-class classification)')

# Plotting the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('plot.png')
