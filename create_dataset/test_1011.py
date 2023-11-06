with open('5k4i.dssp', 'r') as f:
    lines = f.readlines()
seq = ""
p = 0
while lines[p].strip()[0] != "#":
    p += 1
for i in range(p + 1, len(lines)):
    aa = lines[i][13]
    if aa == "!" or aa == "*":
        continue
    seq += aa
print(len(seq))