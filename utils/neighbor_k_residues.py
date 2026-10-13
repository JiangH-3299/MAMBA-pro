import numpy as np
import pandas as pd
import copy

# get pdb coordinates
def get_distance_map(pdb_content):
    cb_coordinates = []
    
    for line in pdb_content.splitlines():
        if line.startswith("ATOM"):
            atom_name = line[12:16].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            
            if atom_name == "CA":
                cb_coordinates.append((x, y, z))
    
    num_residues = len(cb_coordinates)
    distance_map = np.zeros((num_residues, num_residues))
    
    for i in range(int(num_residues)):
        for j in range(i + 1, int(num_residues)):
            residue_i = cb_coordinates[i]
            residue_j = cb_coordinates[j]
            
            distance = np.sqrt(
                (residue_i[0] - residue_j[0]) ** 2 +
                (residue_i[1] - residue_j[1]) ** 2 +
                (residue_i[2] - residue_j[2]) ** 2
            )
            
            distance_map[i][j] = distance
            distance_map[j][i] = distance
    
    return distance_map, num_residues

# get neighbor k residues around mutant residues
def get_k_min(lst, k):
    
    copy_lst = copy.deepcopy(lst)
    min_k_index = []
    for _ in range(k):
        number = min(copy_lst)
        index = copy_lst.index(number)
        min_k_index.append(index)
        copy_lst[index] = 1000
    
    min_k_index.sort()
    
    return min_k_index

def neighbor_k_residues(distance_map, position, k): # from 0 to l-1
    dist_map_df = pd.DataFrame(distance_map)
    
    distance_list = list(dist_map_df.iloc[:, position])
    neighbor_k_pos = get_k_min(distance_list, k+1)
    
    neighbor_k_map = dist_map_df.iloc[neighbor_k_pos, neighbor_k_pos]
    
    return neighbor_k_map, neighbor_k_pos
#output: dataframe, size=[k+1, k+1]; list, size=[k+1]

def get_neighbor_k_res(pdb_path, mut, k=32, full_map=True):
    with open(pdb_path, 'r') as file:
        pdb_content = file.read()
    
    distance_map, num_residues = get_distance_map(pdb_content)
    position = int(mut[1:-1]) # from 0 to l-1
    neighbor_k_map, neighbor_k_pos = neighbor_k_residues(distance_map, position, k)
    neighbor_k_map = np.array(neighbor_k_map)
    
    if full_map == True:
        return neighbor_k_map, neighbor_k_pos, distance_map
    else:
        return neighbor_k_map, neighbor_k_pos