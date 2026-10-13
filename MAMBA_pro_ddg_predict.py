#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAMBA-pro ddg prediction using sequence embeddings and structural features.

Usage:
  python MAMBA_pro_ddg_predict.py --model models/MAMBA_pro_ddg.pkl --pdb-dir PDBs single_seq --pdb 1abc --chain A
  python MAMBA_pro_ddg_predict.py --model models/MAMBA_pro_ddg.pkl multi_seqs --csv mutations.csv
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model import MAMBA_pro
from loader import full_information, load, _AA3_TO_AA1

torch.set_default_dtype(torch.float32)
warnings.filterwarnings('ignore')

# ======================================================================
# PDB helpers
# ======================================================================


def get_chain_sequence(pdb_path, chain_id):
    """Extract amino-acid sequence for a specific chain from a PDB file."""
    seq = []
    seen = set()
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    res_name = line[17:20].strip().upper()
                    chain = line[21:22].strip() if line[21:22].strip() else " "
                    if chain != chain_id:
                        continue
                    res_num = line[22:26].strip()
                    key = (chain, res_num)
                    if key not in seen:
                        seen.add(key)
                        aa = _AA3_TO_AA1.get(res_name, "X")
                        seq.append(aa)
    return "".join(seq)


def list_pdb_chains(pdb_path):
    """Return sorted list of chain IDs found in a PDB file."""
    chains = set()
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                c = line[21:22].strip() if line[21:22].strip() else " "
                chains.add(c)
    return sorted(chains)


# ======================================================================
# Prediction core
# ======================================================================


AA_LIST = "ACDEFGHIKLMNPQRSTVWY"


def _predict_fitness(model, wt_sequence, Id, wt, mut, device, pdb_dir="PDBs", chain=None):
    """Compute features and predict ddG fitness for one mutation."""
    feat = full_information(wt_sequence, Id, wt, mut, pdb_dir=pdb_dir, chain=chain)
    _, _, _, LRP_wt, LRP_mut, GRP_wt, GRP_mut, contact_map, bias, centrality = feat

    LRP_wt = load(LRP_wt, device)
    LRP_mut = load(LRP_mut, device)
    GRP_wt = load(GRP_wt, device)
    GRP_mut = load(GRP_mut, device)
    contact_map = load(contact_map, device)
    bias = load(bias, device)
    centrality = load(centrality, device)

    y_pred = model(LRP_wt, LRP_mut, GRP_wt, GRP_mut, contact_map, bias, centrality)
    fitness = torch.squeeze(y_pred).float().cpu().numpy()
    return float(fitness)


def _make_record(Id, chain, wt, mut, fitness):
    return {"id": Id, "chain": chain, "wt": wt, "mut": mut, "fitness": fitness}


# ======================================================================
# Mode: single_seq  -  full-length saturation mutagenesis
# ======================================================================


def cmd_single_seq(args):
    pdb_path = os.path.join(args.pdb_dir, args.pdb + ".pdb")
    print("Reading WT sequence from", pdb_path)
    wt_sequence = get_chain_sequence(pdb_path, args.chain)

    if len(wt_sequence) == 0:
        avail = list_pdb_chains(pdb_path)
        print("ERROR: No residues found for chain " + repr(args.chain))
        print("  Available chains in PDB: " + str(avail))
        return

    print("WT sequence ({} residues): {}...".format(len(wt_sequence), wt_sequence[:50]))

    device = torch.device(args.device)
    model = MAMBA_pro().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    records = []
    with torch.no_grad():
        for i in tqdm(range(len(wt_sequence)), desc="Scanning positions"):
            wt_aa = wt_sequence[i]
            pos = str(i + 1)
            for mut_aa in (a for a in AA_LIST if a != wt_aa):
                mut = wt_aa + pos + mut_aa
                fitness = _predict_fitness(model, wt_sequence, args.pdb, "wt", mut, device, pdb_dir=args.pdb_dir, chain=args.chain)
                records.append(_make_record(args.pdb, args.chain, "wt", mut, fitness*30.0-15.0))

    result_df = pd.DataFrame(records)
    result_df.to_csv(args.output, index=False)
    print("")
    print("Predictions saved -> " + args.output)
    print("Total: {} mutations".format(len(result_df)))
    print(result_df.head().to_string(index=False))


# ======================================================================
# Mode: multi_seqs  -  predict from CSV with validation
# ======================================================================


def cmd_multi_seqs(args):
    df = pd.read_csv(args.csv)
    required = {"id", "chain", "wt", "mut", "wt_seq"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("CSV missing columns: " + str(missing))

    print("Loaded {} mutations from {}".format(len(df), args.csv))

    device = torch.device(args.device)
    model = MAMBA_pro().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    seq_cache = {}
    records = []
    errors = []

    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
            pdb_id = row["id"]
            chain = row["chain"]
            cache_key = (pdb_id, chain)

            if cache_key not in seq_cache:
                pdb_path = os.path.join(args.pdb_dir, pdb_id + ".pdb")
                pdb_seq = get_chain_sequence(pdb_path, chain)
                if len(pdb_seq) == 0:
                    avail = list_pdb_chains(pdb_path)
                    msg = ("Row " + str(idx) + ": chain " + repr(chain)
                           + " not found in " + pdb_path
                           + ". Available: " + str(avail))
                    print(msg)
                    errors.append(msg)
                    seq_cache[cache_key] = None
                    continue
                seq_cache[cache_key] = pdb_seq

            pdb_seq = seq_cache[cache_key]
            if pdb_seq is None:
                continue

            user_seq = str(row["wt_seq"]).strip()
            if user_seq != pdb_seq:
                msg = ("Row " + str(idx) + ": wt_seq mismatch for " + pdb_id
                       + "/" + chain + ": user len=" + str(len(user_seq))
                       + ", PDB chain len=" + str(len(pdb_seq)))
                print(msg)
                errors.append(msg)
                continue

            fitness = _predict_fitness(model, user_seq, pdb_id, row["wt"], row["mut"], device, pdb_dir=args.pdb_dir, chain=chain)
            records.append(_make_record(pdb_id, chain, row["wt"], row["mut"], fitness*30.0-15.0))

    if errors:
        print("")
        print("WARNING: " + str(len(errors)) + " row(s) skipped.")
        for e in errors[:5]:
            print(e)

    if not records:
        print("No valid predictions generated.")
        return

    result_df = pd.DataFrame(records)
    result_df.to_csv(args.output, index=False)
    print("")
    print("Predictions saved -> " + args.output)
    print("Valid: " + str(len(result_df)) + "  |  Skipped: " + str(len(errors)))
    print(result_df.head().to_string(index=False))


# ======================================================================
# CLI
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="MAMBA-pro ddG prediction - predict binding free energy changes")

    parser.add_argument("--device", default="cuda:0",
                        help="Torch device (default: cuda:0)")
    parser.add_argument("--model", "-m", required=True,
                        help="Path to .pkl model file")
    parser.add_argument("--output", "-o", default="ddg_predictions.csv",
                        help="Output CSV path")
    parser.add_argument("--pdb-dir", "--pdb_dir", default="./PDBs/",
                        help="Directory containing .pdb files (default: ./PDBs/)")

    subparsers = parser.add_subparsers(dest="mode")
    subparsers.required = True

    # ---- single_seq ----
    ss = subparsers.add_parser("single_seq",
         help="Saturation mutagenesis for one PDB chain")
    ss.add_argument("--pdb", "-p", required=True,
                    help="PDB file name (without .pdb suffix)")
    ss.add_argument("--chain", default="A",
                    help="Chain ID (default: A)")

    # ---- multi_seqs ----
    ms = subparsers.add_parser("multi_seqs",
         help="Predict ddG from CSV with id, chain, wt, mut, wt_seq")
    ms.add_argument("--csv", required=True,
                    help="CSV with columns: id, chain, wt, mut, wt_seq")

    args = parser.parse_args()

    if args.mode == "single_seq":
        cmd_single_seq(args)
    else:
        cmd_multi_seqs(args)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()

