from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import utils.neighbor_k_residues as neighbor_k_residues
import utils.graph_preprocess as graph_preprocess
import utils.esm2_embedding as esm2_embedding

# 3-letter to 1-letter amino acid mapping
_AA3_TO_AA1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'HID': 'H', 'HIE': 'H', 'HIP': 'H', 'ASH': 'D', 'GLH': 'E',
    'CYX': 'C', 'MSE': 'M',
}


def get_sequence_from_pdb(pdb_path):
    """Extract wild-type amino acid sequence from a PDB file."""
    seq = []
    seen_residues = set()
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                if atom_name == 'CA':
                    res_name = line[17:20].strip().upper()
                    chain = line[21:22].strip() if line[21:22].strip() else ' '
                    res_num = line[22:26].strip()
                    key = (chain, res_num)
                    if key not in seen_residues:
                        seen_residues.add(key)
                        aa = _AA3_TO_AA1.get(res_name, 'X')
                        seq.append(aa)
    return ''.join(seq)


def get_mutant(wt_seq, mut):
    """Generate a mutant sequence by applying a mutation string to the WT sequence."""
    if mut.startswith('wt'):
        return wt_seq
    else:
        wt_aa = mut[0]
        pos = int(mut[1:-1]) - 1
        mut_aa = mut[-1]

        wt_seq_list = list(wt_seq)
        assert wt_seq_list[pos] == wt_aa, 'Mismatch: ' + mut
        wt_seq_list[pos] = mut_aa
        return ''.join(wt_seq_list)


def embedding_std(embedding):
    """Standardize embedding along axis 1."""
    return (embedding - embedding.mean(axis=1, keepdims=True)) / embedding.std(axis=1, keepdims=True)


def load(arr, device):
    """Convert numpy array to CUDA variable on the specified device."""
    arr = torch.from_numpy(arr).float().unsqueeze(0)
    arr = Variable(arr.to(device))
    return arr


def structure_info(Id, position, pdb_dir="PDBs", chain=None):
    """Extract structure-based features (contact map, bias, centrality) from a PDB file."""
    with open(Path(pdb_dir) / f'{Id}.pdb', 'r') as file:
        pdb_content = file.read()
    if chain is not None:
        pdb_content = '\n'.join(
            line for line in pdb_content.splitlines()
            if line.startswith('ATOM') and (line[21:22].strip() or ' ') == chain
        )
    distance_map, num_residues = neighbor_k_residues.get_distance_map(pdb_content)
    if num_residues <= 33:
        raise ValueError('Feature extraction requires more than 33 CA residues.')
    if not 0 <= position < num_residues:
        raise ValueError(f'Mutation position {position + 1} is outside the structure.')

    neighbor_32_map, neighbor_32_pos = neighbor_k_residues.neighbor_k_residues(distance_map, position, k=32)
    neighbor_32_map = np.array(neighbor_32_map)

    contact_map = graph_preprocess.get_contact_map(neighbor_32_map)
    SPD, centrality = graph_preprocess.compute_SPD_centrality(contact_map)

    bias = SPD
    non_zero_mask = SPD != 0.0
    bias[non_zero_mask] = 1 / bias[non_zero_mask]

    centrality = centrality * 100

    return contact_map, bias, centrality, neighbor_32_pos


def sequence_info(seq, neighbor_32_pos):
    """Extract ESM2 sequence embeddings."""
    GRP = esm2_embedding.get_ESM_embeddings(seq)
    LRP = GRP[neighbor_32_pos]
    return GRP, LRP


def full_information(wt_sequence, Id, wt, mut, pdb_dir="PDBs", chain=None):
    """Compute all features (structure + sequence) for a given mutation pair."""
    wt_seq = get_mutant(wt_sequence, wt)
    mut_seq = get_mutant(wt_sequence, mut)

    if wt.startswith('wt'):
        if mut.startswith('wt'):
            position = 0
        else:
            position = int(mut[1:-1]) - 1
    else:
        position = int(wt[1:-1]) - 1

    contact_map, bias, centrality, neighbor_32_pos = structure_info(Id, position, pdb_dir=pdb_dir, chain=chain)

    GRP_wt, LRP_wt = sequence_info(wt_seq, neighbor_32_pos)
    length = len(GRP_wt)
    LRP_wt = embedding_std(LRP_wt)
    GRP_wt = embedding_std(GRP_wt)
    GRP_wt = (GRP_wt.sum(axis=0) - LRP_wt.mean(axis=0)) / (length - 33)
    GRP_wt = (GRP_wt - GRP_wt.mean()) / (GRP_wt.std() + 1e-8)

    GRP_mut, LRP_mut = sequence_info(mut_seq, neighbor_32_pos)
    LRP_mut = embedding_std(LRP_mut)
    GRP_mut = embedding_std(GRP_mut)
    GRP_mut = (GRP_mut.mean(axis=0) - LRP_mut.mean(axis=0)) / (length - 33)
    GRP_mut = (GRP_mut - GRP_mut.mean()) / (GRP_mut.std() + 1e-8)

    return Id, wt, mut, LRP_wt, LRP_mut, GRP_wt, GRP_mut, contact_map, bias, centrality

class ProDataset(Dataset):
    def __init__(self, dataframe, wt_sequence):
        self.Ids = dataframe['id'].values
        self.wts = dataframe['wt'].values
        self.muts = dataframe['mut'].values
        self.labels = dataframe['label'].values
        self.teachers = dataframe['teacher'].values
        self.wt_seq = wt_sequence
    
    def __getitem__(self, index):
        Id = self.Ids[index]
        wt = self.wts[index]
        mut = self.muts[index]
        label = self.labels[index]
        teacher = self.teachers[index]
        wt_seq = self.wt_seq
        
        Id, wt, mut, LRP_wt, LRP_mut, GRP_wt, GRP_mut, contact_map, bias, centrality = full_information(self.wt_seq, Id, wt, mut)
        
        return Id, wt, mut, LRP_wt, LRP_mut, GRP_wt, GRP_mut, contact_map, bias, centrality, label, teacher

    def __len__(self):
        return len(self.labels)
