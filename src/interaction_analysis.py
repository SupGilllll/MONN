import numpy as np
from sklearn import metrics

records = np.load('/data/zhao/MONN/results/240116/transformer/load_model/records.npy', allow_pickle=True)
baseline_records = np.load('/data/zhao/MONN/results/240116/transformer/load_model/baseline_records.npy', allow_pickle=True).item()
sorted_records = sorted(records, key=lambda x: x['auc'], reverse=True)
top_5_records = sorted_records[:5]
record = top_5_records[0]
print(np.count_nonzero(record['int_label']))
for i, record in enumerate(sorted_records[:200]):
    print(i, record['pdbid'], np.count_nonzero(record['int_label']), record['auc'], record['int_label'].shape, baseline_records[record['pdbid']]['auc'])

# 4x3t 39 0.9663331019263222 (29, 158) 9th element
# record = sorted_records[8]
# fpr, tpr, thresholds = metrics.roc_curve(record['int_label'].reshape(-1), record['int_pred'].reshape(-1))
# print(metrics.auc(fpr, tpr))
# optimal_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))
# optimal_threshold = thresholds[optimal_idx]

# print(thresholds)
# print(f"Optimal threshold is: {optimal_threshold}")