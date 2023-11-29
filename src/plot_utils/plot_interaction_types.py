import matplotlib.pyplot as plt
import seaborn as sns

# Data to be visualized
# *** 2023/11/20 results ***
# data = {
#     'Hydrogen Bonds': 58289, 'Salt Bridges': 25141, 'Water Bridges': 18426, 
#     'Water Bridges + Salt Bridges': 2545, 'Hydrophobic Interactions': 72915, 
#     'Hydrogen Bonds + Salt Bridges': 509, 'pi-Stacking': 38566, 
#     'Halogen Bonds': 1338, 'pi-Cation Interactions': 8012, 'Hydrogen Bonds + pi-Cation Interactions': 141, 
#     'Hydrogen Bonds + Water Bridges': 2634, 'Hydrophobic Interactions + pi-Stacking': 1644, 
#     'Hydrophobic Interactions + pi-Cation Interactions': 300, 'Water Bridges + pi-Cation Interactions': 58, 
#     'pi-Stacking + pi-Cation Interactions': 547, 'Water Bridges + pi-Stacking': 69, 
#     'Hydrogen Bonds + Water Bridges + Salt Bridges': 51, 'Hydrogen Bonds + pi-Stacking + pi-Cation Interactions': 3, 
#     'Hydrogen Bonds + pi-Stacking': 64, 'pi-Cation Interactions + Salt Bridges': 4, 
#     'Hydrophobic Interactions + pi-Stacking + pi-Cation Interactions': 8, 'pi-Stacking + Salt Bridges': 3, 
#     'Water Bridges + pi-Stacking + pi-Cation Interactions': 11, 'Hydrogen Bonds + Water Bridges + pi-Cation Interactions': 4, 
#     'Non-interaction': 262951371
# }
# *** 2023/12/04 results ***
# data = {'Hydrogen Bonds': 58269, 'Salt Bridges': 25134, 'Water Bridges': 18421, 
#         'Water Bridges_Salt Bridges': 2543, 'Hydrophobic Interactions': 72905, 
#         'Hydrogen Bonds_Salt Bridges': 509, 'pi-Stacking': 38560, 'Halogen Bonds': 1338, 
#         'pi-Cation Interactions': 8012, 'Hydrogen Bonds_pi-Cation Interactions': 141, 
#         'Hydrogen Bonds_Water Bridges': 2634, 'Hydrophobic Interactions_pi-Stacking': 1644, 
#         'Hydrophobic Interactions_pi-Cation Interactions': 300, 'Water Bridges_pi-Cation Interactions': 58, 
#         'pi-Stacking_pi-Cation Interactions': 547, 'Water Bridges_pi-Stacking': 69, 
#         'Hydrogen Bonds_Water Bridges_Salt Bridges': 51, 'Hydrogen Bonds_pi-Stacking_pi-Cation Interactions': 3, 
#         'Hydrogen Bonds_pi-Stacking': 64, 'pi-Cation Interactions_Salt Bridges': 4, 
#         'Hydrophobic Interactions_pi-Stacking_pi-Cation Interactions': 8, 
#         'pi-Stacking_Salt Bridges': 3, 'Water Bridges_pi-Stacking_pi-Cation Interactions': 11, 
#         'Hydrogen Bonds_Water Bridges_pi-Cation Interactions': 4, 'Non-interaction': 262924582}
data = {'Non-interaction': 262924582, 'Hydrogen Bonds': 59068, 'Water Bridges': 20375, 
        'Hydrophobic Interactions': 73285, 'pi-Stacking': 40175, 'pi-Cation Interactions': 8846, 
        'Salt Bridges': 28145, 'Halogen Bonds': 1338}


sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: item[1])}

plt.figure(figsize=(15, 8))
sns_barplot = sns.barplot(x=list(sorted_data.values()), y=list(sorted_data.keys()), color="lightblue")
plt.xlabel('Count (Log Scale)')
plt.xscale('log')  # Keeping the logarithmic scale for x-axis
plt.title('Counts of Various Interactions')

# Adding the count data on the bars
for p in sns_barplot.patches:
    width = p.get_width()  # Get the width of each bar
    plt.text(width,  # Set the text at the width of the bar
             p.get_y() + p.get_height() / 2,  # Set text at the middle height of the bar
             '{}'.format(int(width)),  # The count (formatted to 2 decimal places)
             ha='right',  # Horizontal alignment
             va='center')  # Vertical alignment

plt.savefig('plot.png', bbox_inches = 'tight')

