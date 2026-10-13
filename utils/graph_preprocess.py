import numpy as np
import networkx as nx

# from distance_map to contact_map
def get_contact_map(graph_map, threshold=12.0):
    length = len(graph_map)
    contact_map = np.zeros((length, length), dtype=int)
    
    for i in range(length):
        contact_map[i][i] = 1
        for j in range(i+1, length):
            distance = graph_map[i][j]
            if distance <= threshold:
                contact_map[i][j] = 1
                contact_map[j][i] = 1
    
    return contact_map

# get SPD matrix and node centrality
def compute_SPD_centrality(contact_map):
    shape = contact_map.shape
    graph = nx.from_numpy_array(contact_map)
    centrality = np.array(list(nx.centrality.betweenness_centrality(graph).values()))
    SPD = np.empty(shape=shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            try:
                SPD[i][j] = nx.shortest_path_length(graph, source=i, target=j)
            except:
                SPD[i][j] = 0.0
    
    return SPD.astype(int), centrality