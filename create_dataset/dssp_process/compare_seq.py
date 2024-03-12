import pickle
import os

with open('../out4_interaction_dict', 'rb') as f:
    data = pickle.load(f)

def process_dssp(pid):
    seq_dict = {}
    id_dict = {}
    with open(f'../dssp_files/{pid}.dssp', "r") as f:
        lines = f.readlines()
    p = 0
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        str = lines[i].strip().split()
        aa = lines[i][13]
        if aa == "!" or aa == "*" or aa == 'X':
            continue
        if aa.islower():
            aa = 'C'
        chain = lines[i][11]
        try:
            idx = int(lines[i][7:11].strip())
        except:
            continue
        if chain not in id_dict:
            id_dict[chain] = []
        id_dict[chain].append(idx)
        if chain not in seq_dict:
            seq_dict[chain] = ''
        seq_dict[chain] += aa
    return seq_dict, id_dict
    
def find_special_char(s1, s2, chain):
    print(len(s1), len(s2))
    # print(s1)
    # print(s2)
    # assert len(s1) == len(s2)
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            print(chain, i, s1[i], s2[i])
            break

def find_special_idx(s1, s2, chain):
    print(len(s1), len(s2))
    # print(s1)
    # print(s2)
    # assert len(s1) == len(s2)
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            print(chain, i, s1[i], s2[i])
            break

def compare_dssp(element):
    pid = element.split('_')[0]
    assert len(pid) == 4
    sample = data[element]['sequence']
    chains = data[element]['sequence'].keys()
    seq_dict, id_dict = process_dssp(pid)
    for chain in chains:
        if len(sample[chain][0]) == 0:
            continue
        assert len(seq_dict[chain]) == len(id_dict[chain])
        if seq_dict[chain] != sample[chain][0]:
            print('seq not equal', element, chain)
            find_special_char(seq_dict[chain], sample[chain][0], chain)
            return False
        if id_dict[chain] != sample[chain][1]:
            print('idx not equal', element, chain)
            find_special_idx(id_dict[chain], sample[chain][1], chain)
            return False
    return True

i = 0
false_cnt = 0
for element in data.keys():
    print(element)
    if i % 100 == 0:
        print(f"------------process {i} samples------------")
    if not compare_dssp(element):
        false_cnt += 1
    # compare_dssp(element)
    # if not compare_dssp(element):
    #     print("not equal ", element)
    i += 1

# print(compare_dssp('2j9n_BEN'))
        
print("false_cnt:", false_cnt)
