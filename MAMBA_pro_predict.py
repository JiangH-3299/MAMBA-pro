import warnings
import numpy as np
import pandas as pd
import torch
import multiprocessing
import argparse
import os
from tqdm import tqdm

from model import MAMBA_pro
from loader import get_sequence_from_pdb, load, full_information

torch.set_default_dtype(torch.float32)
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)


# ------------------------------------------------------------------------------
# Prediction core: shared by both modes

def _predict_single(model, wt_sequence, Id, wt, mut, device):
    Id, wt, mut, LRP_wt, LRP_mut, GRP_wt, GRP_mut, contact_map, bias, centrality = \
        full_information(wt_sequence, Id, wt, mut)

    LRP_wt = load(LRP_wt, device)
    LRP_mut = load(LRP_mut, device)
    GRP_wt = load(GRP_wt, device)
    GRP_mut = load(GRP_mut, device)
    contact_map = load(contact_map, device)
    bias = load(bias, device)
    centrality = load(centrality, device)

    y_pred = model(LRP_wt, LRP_mut, GRP_wt, GRP_mut, contact_map, bias, centrality)
    y_pred = torch.squeeze(y_pred).float().cpu().numpy()
    return y_pred


def run_full_length(model, wt_sequence, Id, device):
    model.eval()
    records = []

    with torch.no_grad():
        print("\n========== Predicting full length ==========")
        for i in tqdm(range(len(wt_sequence))):
            wt_aa = wt_sequence[i]
            pos = str(i + 1)
            for mut_aa in (aa for aa in 'ACDEFGHIKLMNPQRSTVWY' if aa != wt_aa):
                mut = wt_aa + pos + mut_aa
                y_pred = _predict_single(model, wt_sequence, Id, 'wt', mut, device)
                records.append({'id': Id, 'wt': 'wt', 'mut': mut, 'fitness': y_pred})

    return pd.DataFrame(records)


def run_selected_sites(model, wt_sequence, dataframe, device):
    model.eval()
    records = []

    with torch.no_grad():
        print("\n========== Predicting selected sites ==========")
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            y_pred = _predict_single(model, wt_sequence, row['id'], row['wt'], row['mut'], device)
            records.append({'id': row['id'], 'wt': row['wt'], 'mut': row['mut'], 'fitness': y_pred})

    return pd.DataFrame(records)


# ------------------------------------------------------------------------------
# CLI entry point

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='MAMBA-pro Predict (with seq/PDB length validation)')
    parser.add_argument('--device', default='cuda:0', help='Device (e.g., cuda:0, cpu)')
    parser.add_argument('--model', '-m', required=True, help='Path to pretrained model (.pkl)')
    parser.add_argument('--pdb', '-p', required=True, help='PDB file name (without .pdb suffix, in PDBs/)')
    parser.add_argument('--seq', '-s', required=True, help='WT sequence (required, will be validated against PDB length)')
    parser.add_argument('--output', '-o', default='MAMBA_pro_pred.csv', help='Output CSV path')

    subparsers = parser.add_subparsers(dest='mode', required=True, help='Prediction mode')

    # full-length mode
    full_parser = subparsers.add_parser('full', help='Full-length scanning')

    # selected-sites mode
    site_parser = subparsers.add_parser('sites', help='Predict selected mutations')
    site_parser.add_argument('--csv', required=True, help='CSV file with columns: id, wt, mut')

    args = parser.parse_args()
    device = torch.device(args.device)

    # --- Validate input seq length against PDB structure ---
    pdb_path = os.path.join('PDBs', args.pdb + '.pdb')
    pdb_seq = get_sequence_from_pdb(pdb_path)
    if len(args.seq) != len(pdb_seq):
        raise ValueError(
            f"Sequence length mismatch: input seq has {len(args.seq)} residues, "
            f"but PDB structure ({args.pdb}) has {len(pdb_seq)} residues."
        )
    wt_sequence = args.seq
    print(f"PDB structure has {len(pdb_seq)} residues, input seq matches.")
    print(f"WT sequence ({len(wt_sequence)} residues): {wt_sequence}")

    # Build model and load weights
    model = MAMBA_pro().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    if args.mode == 'full':
        result_df = run_full_length(model, wt_sequence, args.pdb, device)
    else:
        dataframe = pd.read_csv(args.csv)
        result_df = run_selected_sites(model, wt_sequence, dataframe, device)

    result_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
