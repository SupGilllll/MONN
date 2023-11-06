import os 
import re

os.chdir('/data/zhao/MONN/create_dataset')
def get_pdbid_list():
	pdbid_list = []
	with open('./out2_pdbbind_all_datafile.tsv') as f:
		for line in f.readlines():
			pdbid_list.append(line.strip().split('\t')[0])
	print('pdbid_list',len(pdbid_list))
	return pdbid_list

pdbid_list = get_pdbid_list()
with open('./wget_dssp.txt', 'w') as f:
	for pdbid in pdbid_list:
	    f.write(f'https://pdb-redo.eu/dssp/get?pdb-id={pdbid}&format=dssp&_csrf=iIXTmmOAWeiz2vNhVh0CKg\n')

# Define the directory where your files are located
# directory = "./dssp_files"

# # Define a regular expression pattern to match the pdbid
# pattern = r'pdb-id=([A-Za-z0-9]+)'

# # List all files in the directory
# files = os.listdir(directory)

# # Iterate through the files and rename them
# for file in files:
#     # Use regular expression to extract the pdbid
#     match = re.search(pattern, file)
#     if match:
#         pdbid = match.group(1)
#         new_name = os.path.join(directory, f'{pdbid}.dssp')
#         old_path = os.path.join(directory, file)
        
#         # Rename the file
#         os.rename(old_path, new_name)
#         print(f'Renamed {file} to {pdbid}.dssp')
	