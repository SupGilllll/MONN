# 1. calculate the proportion of interaction sites in pocket area
# 2. calculate the proportion of pocket area in whole surface area
# @ calculate based on original data rather than alignment
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

def density_plot(prop_list1, prop_list2, prop_list3):
    # sns.displot(prop_list, kind="kde")
    plt.rcParams.update({'font.size': 18}) 
    sns.displot(prop_list1, label='Ligand binding sites in pockets')
    plt.xlabel('Proportion')
    plt.ylabel('Count')
    plt.title('Distribution Plot')
    plt.legend()
    plt.savefig('plot1.png', bbox_inches='tight')

    plt.cla()
    sns.displot(prop_list2, label='Pockets in surface residues')
    plt.xlabel('Proportion')
    plt.ylabel('Count')
    plt.title('Distribution Plot')
    plt.legend()
    plt.savefig('plot2.png', bbox_inches='tight')

    plt.cla()
    sns.displot(prop_list3, label='Ligand binding sites in surface residues')
    plt.xlabel('Proportion')
    plt.ylabel('Count')
    plt.title('Distribution Plot')
    plt.legend()
    plt.savefig('plot3.png', bbox_inches='tight')

def calculate_intsites_in_pocket(cid_list):
    prop_list = []
    for cid in cid_list:
        pid = cid.split('_')[0]
        int_cnt = len(int_dict[cid]['residue_interact'])
        hit_cnt = 0
        seq1 = ''
        seq2 = ''
        for item in int_dict[cid]['residue_interact']:
            chain, idx, aa = item[0][0], item[0][1:], item[1]
            idx += ' '
            if chain in pocket_dict[pid]['pocket'] and idx in pocket_dict[pid]['pocket'][chain][1]:
                seq1 += aa
                idx2 = pocket_dict[pid]['pocket'][chain][1].index(idx)
                seq2 += pocket_dict[pid]['pocket'][chain][0][idx2]
                hit_cnt += 1
        if seq1 == seq2:
            prop_list.append(hit_cnt / int_cnt)
    return prop_list

def calculate_pocket_in_surface(cid_list):
    prop_list = []
    for cid in cid_list:
        pid = cid.split('_')[0]
        pocket_cnt = 0
        hit_cnt = 0
        seq1 = ''
        seq2 = ''
        for chain in pocket_dict[pid]['pocket'].keys():
            pocket_cnt += len(pocket_dict[pid]['pocket'][chain][1])
            for idx in pocket_dict[pid]['pocket'][chain][1]:
                real_idx = int(''.join(filter(str.isdigit, idx)))
                if chain in surface_dict[cid]['surface'] and real_idx in surface_dict[cid]['surface'][chain][1]:
                    idx1 = pocket_dict[pid]['pocket'][chain][1].index(idx)
                    seq1 += pocket_dict[pid]['pocket'][chain][0][idx1]
                    idx2 = surface_dict[cid]['surface'][chain][1].index(real_idx)
                    seq2 += surface_dict[cid]['surface'][chain][0][idx2]
                    hit_cnt += 1
        if seq1 == seq2:
            prop_list.append(hit_cnt / pocket_cnt)
    return prop_list

def calculate_intsites_in_surface(cid_list):
    prop_list = []
    for cid in cid_list:
        int_cnt = len(int_dict[cid]['residue_interact'])
        hit_cnt = 0
        seq1 = ''
        seq2 = ''
        for item in int_dict[cid]['residue_interact']:
            chain, idx, aa = item[0][0], int(item[0][1:]), item[1]
            if chain in surface_dict[cid]['surface'] and idx in surface_dict[cid]['surface'][chain][1]:
                seq1 += aa
                idx2 = surface_dict[cid]['surface'][chain][1].index(idx)
                seq2 += surface_dict[cid]['surface'][chain][0][idx2]
                hit_cnt += 1
        if seq1 == seq2:
            prop_list.append(hit_cnt / int_cnt)
    return prop_list

with open('../out3_surface_dict', 'rb') as f:
    surface_dict = pickle.load(f)
with open('../out4_interaction_dict', 'rb') as f:
    int_dict = pickle.load(f)
with open('../out5_pocket_dict', 'rb') as f:
    pocket_dict = pickle.load(f)

cid_list = list(int_dict.keys())
intsites_prop_list = calculate_intsites_in_pocket(cid_list)
pocket_prop_list = calculate_pocket_in_surface(cid_list)
sites_surface_prop_list = calculate_intsites_in_surface(cid_list)
print(len(intsites_prop_list), len(pocket_prop_list), len(sites_surface_prop_list))
density_plot(intsites_prop_list, pocket_prop_list, sites_surface_prop_list)
print("------------finish processing!------------")
