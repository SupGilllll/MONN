import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rcParams.update({'font.size': 12}) 
# Data from the classification report (manually inputted from the image provided by the user)
# data = {
#     'Class': [
#         'Non-interaction', 'Hydrogen Bonds', 'Water Bridges', 'Hydrophobic Interactions',
#         'pi-Stacking', 'pi-Cation Interactions', 'Salt Bridges', 'Halogen Bonds'
#     ],
#     'Precision': [1.00, 0.001, 0.00, 0.001, 0.003, 0.001, 0.008, 0.00],
#     'Recall': [0.513, 0.436, 0.536, 0.732, 0.895, 0.880, 0.987, 0.995],
#     'F1-Score': [0.678, 0.003, 0.001, 0.003, 0.007, 0.001, 0.016, 0.001]
# }

data = {
    'Class': [
        'Hydrogen Bonds', 'Water Bridges', 'Hydrophobic Interactions',
        'pi-Stacking', 'pi-Cation Interactions', 'Salt Bridges', 'Halogen Bonds'
    ],
    'Precision': [0.795, 0.265, 0.987, 0.746, 0.528, 0.903, 0.998],
    'Recall': [0.447, 0.557, 0.797, 0.899, 0.894, 0.990, 1.000],
    'F1-Score': [0.572, 0.359, 0.882, 0.815, 0.664, 0.944, 0.999]
}

# data = {
#     'Class': [
#         'Non-interaction', 'Hydrogen Bonds', 'Water Bridges', 'Hydrophobic Interactions',
#         'pi-Stacking', 'pi-Cation Interactions', 'Salt Bridges', 'Halogen Bonds'
#     ],
#     'Precision': [1.00, 0.001, 0.00, 0.001, 0.003, 0.001, 0.007, 0.00],
#     'Recall': [0.513, 0.436, 0.536, 0.732, 0.895, 0.880, 0.987, 0.995],
#     'F1-Score': [0.678, 0.003, 0.001, 0.003, 0.007, 0.001, 0.016, 0.001]
# }

# Convert the data to a DataFrame
df = pd.DataFrame(data).set_index('Class')

# Generate a heatmap
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Score'})
plt.title('Classification Report (threshold = 0.4)')
plt.savefig('plot.png', bbox_inches = 'tight')