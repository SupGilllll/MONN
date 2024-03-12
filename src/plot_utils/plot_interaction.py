import matplotlib.pyplot as plt
import numpy as np

# Example data
protein_residues = ['Residue ' + str(i) for i in range(1, 100)]  # 10 protein residues
compound_atoms = ['Atom ' + str(i) for i in range(1, 6)]        # 5 compound atoms

# Generating random interaction data (1 for interaction, 0 for no interaction)
np.random.seed(0)
interaction_data = np.random.randint(0, 2, size=(len(protein_residues), len(compound_atoms)))

# Plotting
fig, ax = plt.subplots(figsize=(8, 14))

# Draw lines for protein residues and compound atoms
ax.scatter(np.zeros(len(protein_residues)), np.arange(len(protein_residues)), marker='o', s=1, label='Protein Residues')
ax.scatter(np.ones(len(compound_atoms)), np.arange(len(compound_atoms)), marker='s', s=1, label='Compound Atoms')

# # Add labels for protein residues and compound atoms
# for i, label in enumerate(protein_residues):
#     ax.text(-0.2, i, label, horizontalalignment='right')

# for i, label in enumerate(compound_atoms):
#     ax.text(1.2, i, label, horizontalalignment='left')

# Connect interactions
for i in range(len(protein_residues))[:40]:
    for j in range(len(compound_atoms)):
        if interaction_data[i, j] == 1:
            ax.plot([0, 1], [i, j], 'k-')  # 'k-' specifies a black solid line

# Set limits and labels
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-1, max(len(protein_residues), len(compound_atoms)))
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Protein-Compound Interaction Visualization")
ax.legend(loc='upper right')

plt.savefig('plot.png', bbox_inches = 'tight')
