import os
import pickle

import numpy as np

DSSP = '/gs/hs0/tga-ishidalab/zhao/DSSP/dssp-3.1.4/mkdssp'


def process_dssp(dssp_file):
    aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC"
    rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

    with open(dssp_file, "r") as f:
        lines = f.readlines()

    seq = ""
    dssp_feature = []

    p = 0
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        aa = lines[i][13]
        if aa == "!" or aa == "*" or aa == '0':
            continue
        seq += aa
        SS = lines[i][16]
        if SS == " ":
            SS = "C"
        SS_vec = np.zeros(9)  # The last dim represents "Unknown" for missing residues
        SS_vec[SS_type.find(SS)] = 1
        PHI = float(lines[i][103:109].strip())
        PSI = float(lines[i][109:115].strip())
        ACC = float(lines[i][34:38].strip())
        ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
        dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))

    return seq, np.array(dssp_feature)


def pad_dssp(seq, feature, ref_seq):  # ref_seq is longer
    padded_feature = []
    SS_vec = np.zeros(9)  # The last dim represent "Unknown" for missing residues
    SS_vec[-1] = 1
    padded_item = np.concatenate((np.array([360, 360, 0]), SS_vec))

    p_ref = 0
    for i in range(len(seq)):
        while p_ref < len(ref_seq) and seq[i] != ref_seq[p_ref]:
            padded_feature.append(padded_item)
            p_ref += 1
        if p_ref < len(ref_seq):  # aa matched
            padded_feature.append(feature[i])
            p_ref += 1
        else:  # miss match!
            return np.array([])

    if len(padded_feature) != len(ref_seq):
        for i in range(len(ref_seq) - len(padded_feature)):
            padded_feature.append(padded_item)

    return np.array(padded_feature)


def transform_dssp(dssp_feature):
    angle = dssp_feature[:, 0:2]
    ASA_SS = dssp_feature[:, 2:]

    radian = angle * (np.pi / 180)
    dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis=1)

    return dssp_feature


def get_dssp(ID, PDB_seq):
    os.system(f'{DSSP} -i ../Feature/Proteins/pdb/{ID}.pdb -o dssp/{ID}.dssp')
    dssp_seq, dssp_matrix = process_dssp("dssp/" + ID + ".dssp")
    if len(dssp_seq) > len(PDB_seq):
        return "DSSP too long"
    elif len(dssp_seq) < len(PDB_seq):
        padded_dssp_matrix = pad_dssp(dssp_seq, dssp_matrix, PDB_seq)
        if len(padded_dssp_matrix) == 0:
            return "Fail to pad DSSP"
        else:
            np.save("processed_dssp/" + ID, transform_dssp(padded_dssp_matrix))
    else:
        np.save("processed_dssp/" + ID, transform_dssp(dssp_matrix))
    return 0


with open('../Models/Proteins_Dict.pkl', 'rb') as f:
    p_dict = pickle.load(f)
for key, value in p_dict.items():
    get_dssp(value, key)
