# 1. calculate the proportion of pocket area in whole sequence
# @ pockey area calculation based on complex PDB file from rcsb
# 2. calculate the proportion of interaction sites in whole sequence
# 3. calculate the proportion of surface area in whole sequence
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# def calculate_pdbbind(cid_list):
#     prop_list = []
#     for cid in cid_list:
#         pid = cid.split('_')[0]
#         res_cnt = 0
#         hit_cnt = 0
#         for _, data in pocket_dict[pid]['protein'].items():
#             res_cnt += len(data[1])
#         for _, data in pocket_dict[pid]['pocket'].items():
#             hit_cnt += len(data[1])
#         prop_list.append(hit_cnt / res_cnt)
#     return prop_list

def density_plot(prop_list1, prop_list2, prop_list3):
    # sns.displot(prop_list, kind="kde")
    sns.color_palette("deep")
    sns.displot(prop_list1, label='Ligand binding sites in protein sequences', color=sns.xkcd_rgb['nice blue'])
    plt.xlabel('Proportion')
    plt.ylabel('Count')
    plt.title('Distribution plot')
    plt.legend()
    plt.savefig('plot1.png', bbox_inches='tight')

    plt.cla()
    sns.displot(prop_list2, label='Pockets in protein sequences', color=sns.xkcd_rgb['nice blue'])
    plt.xlabel('Proportion')
    plt.ylabel('Count')
    plt.title('Distribution plot')
    plt.legend()
    plt.savefig('plot2.png', bbox_inches='tight')

    plt.cla()
    sns.displot(prop_list3, label='Surface residues in protein sequences', color=sns.xkcd_rgb['nice blue'])
    plt.xlabel('Proportion')
    plt.ylabel('Count')
    plt.title('Distribution plot')
    plt.legend()
    plt.savefig('plot3.png', bbox_inches='tight')
    

def calculate_pocket(cid_list):
    prop_list = []
    for cid in cid_list:
        pid = cid.split('_')[0]
        res_cnt = 0
        hit_cnt = 0
        for data in int_dict[cid]['sequence'].values():
            res_cnt += len(data[1])
        for data in pocket_dict[pid]['pocket'].values():
            hit_cnt += len(data[1])
        prop_list.append(hit_cnt / res_cnt)
    return prop_list

def calculate_intsites(cid_list):
    prop_list = []
    for cid in cid_list:
        res_cnt = 0
        hit_cnt = 0
        for data in int_dict[cid]['sequence'].values():
            res_cnt += len(data[1])
        hit_cnt = len(int_dict[cid]['residue_interact'])
        prop_list.append(hit_cnt / res_cnt)
    return prop_list

def calculate_surface(cid_list):
    prop_list = []
    for cid in cid_list:
        res_cnt = 0
        hit_cnt = 0
        for data in int_dict[cid]['sequence'].values():
            res_cnt += len(data[1])
        for data in surface_dict[cid]['surface'].values():
            hit_cnt += len(data[1])
        prop_list.append(hit_cnt / res_cnt)
    return prop_list

with open('../out4_interaction_dict', 'rb') as f:
    int_dict = pickle.load(f)
with open('../out5_pocket_dict', 'rb') as f:
    pocket_dict = pickle.load(f)
with open('../out3_surface_dict', 'rb') as f:
    surface_dict = pickle.load(f)

cid_list = list(int_dict.keys())
intsites_prot_list = calculate_intsites(cid_list)
pocket_prot_list = calculate_pocket(cid_list)
surface_prot_list = calculate_surface(cid_list)
density_plot(intsites_prot_list, pocket_prot_list, surface_prot_list)
print("------------finish processing!------------")
