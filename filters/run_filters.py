import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import multiprocessing
from tqdm import tqdm
import os

from utils.GA import GraphAttention
from utils.GCN import GCN

import utils.neighbor_k_residues as neighbor_k_residues
import utils.graph_preprocess as graph_preprocess
import utils.esm2_embedding as esm2_embedding

from loader import full_information, load

torch.set_default_dtype(torch.float32)
warnings.filterwarnings('ignore')

# Seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    device = torch.device('cuda:1' if torch.cuda.device_count() >= 2 else 'cuda:0')
    torch.cuda.set_device(device)
except RuntimeError:
    device = torch.device('cuda:0')


# parameters
LENG_SIZE = 1000
GCN_input_dim = 1280
GCN_hidden_dim = 256
GCN_output_dim = 63
Attention_output_dim = 64
Num_classes = 1

# Model -----------------------------------------------------------------------

# define training model
class Model(nn.Module):
    def __init__(self, num_heads = 8):
        super(Model, self).__init__()
        
        self.gcn = GCN(GCN_input_dim, GCN_hidden_dim, GCN_output_dim)
        self.graphormer = GraphAttention()
        
        self.GRP_linear = nn.Linear(1280, 64)
        
        self.LN1 = nn.LayerNorm(Attention_output_dim)
        self.LN2 = nn.LayerNorm(Attention_output_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1))
    
    def PSSR_module(self, x, adj, attn_bias, centrality):
        ## input x size is [bsz, 33, 1280]
        padding_mask = x[:, :, 0].eq(0) #[bsz, 33]
        
        # gcn module
        x = self.gcn(x, adj) #[bsz, 33, 63]
        
        # graphormer module
        x = torch.cat((x, centrality.unsqueeze(-1)), dim=-1)  #[bsz, 33, 64]
        x = x.masked_fill(padding_mask.unsqueeze(2), float(0))
        x, attn_weight = self.graphormer(x, attn_bias, padding_mask) #[bsz, 33, 64]
        
        return x, attn_weight
    
    def forward(self, x1, x2, x3, x4, adj, attn_bias, centrality):
        ## input x1, X2 size are [bsz, 33, 1280], x1 = LRP_wt, x2 = LRP_mut
        ## input x3, x4 size are [bsz, 1280], x3 = GRP_wt, x4 = GRP_mut

        # LPR: PSSR module
        x1, _ = self.PSSR_module(x1, adj, attn_bias, centrality) #[bsz, 33, 64]
        x2, _ = self.PSSR_module(x2, adj, attn_bias, centrality) #[bsz, 33, 64]
        
        # GRP: no process
        GRP = x4 - x3 # [bsz, 1280]
        GRP = self.GRP_linear(GRP) # [bsz, 64]
        
        # Fusion module: combine LRP and GRP
        x1 = torch.mean(x1, dim=1)
        x1 = self.LN1(x1) #[bsz, 64]
        x2 = torch.mean(x2, dim=1)
        x2 = self.LN2(x2) #[bsz, 64]
        
        LRP = x2 - x1 # [bsz, 64]
        x = torch.cat((LRP, GRP), dim=-1) # [bsz, 128]
        
        output = self.classifier(x)

        return output.squeeze(-1)

#------------------------------------------------------------------------------

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
            for mut_aa in (aa for aa in "ACDEFGHIKLMNPQRSTVWY" if aa != wt_aa):
                mut = wt_aa + pos + mut_aa
                logits = _predict_single(model, wt_sequence, Id, "wt", mut, device)
                y_pred = torch.sigmoid(torch.tensor(logits)).float().numpy()
                y_class = (y_pred > 0.5).astype(float)
                records.append({"id": Id, "wt": "wt", "mut": mut, "pred": y_pred, "class": y_class})

    return pd.DataFrame(records)

def run_selected_sites(model, wt_sequence, dataframe, device):
    model.eval()
    records = []

    with torch.no_grad():
        print("\n========== Predicting selected sites ==========")
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            logits = _predict_single(model, wt_sequence, row["id"], row["wt"], row["mut"], device)
            y_pred = torch.sigmoid(torch.tensor(logits)).float().numpy()
            y_class = (y_pred > 0.5).astype(float)
            records.append({"id": row["id"], "wt": row["wt"], "mut": row["mut"], "pred": y_pred, "class": y_class})

    return pd.DataFrame(records)


# ------------------------------------------------------------------------------
# CLI: batch prediction with all models from saved_model folder

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    import sys, argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Batch prediction with all models from saved_model folder")
    parser.add_argument("--saved_model", default="./saved_model",
                        help="Folder containing .pkl model files (default: ./saved_model)")
    parser.add_argument("--pdb", "-p", required=True, help="PDB file name (without .pdb suffix)")
    parser.add_argument("--seq", "-s", required=True, help="WT sequence")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    parser.add_argument("--device", default="cuda:0", help="Device (e.g. cuda:0, cpu)")
    parser.add_argument("--mode", required=True, choices=["full", "sites"],
                        help="full: scan all single mutations | sites: predict from CSV")
    parser.add_argument("--csv", help="Required for sites mode: CSV with columns id, wt, mut")

    args = parser.parse_args()
    device = torch.device(args.device)

    if args.mode == "sites" and not args.csv:
        parser.error("--csv is required when --mode=sites")

    # Scan for model files
    model_dir = Path(args.saved_model)
    if not model_dir.exists():
        print(f"ERROR: {model_dir} does not exist"); sys.exit(1)
    model_paths = sorted(model_dir.glob("*.pkl"))
    if not model_paths:
        print(f"No .pkl files found in {model_dir}"); sys.exit(1)
    print(f"Found {len(model_paths)} model(s): {[p.name for p in model_paths]}")

    # Load input data for sites mode once
    input_df = pd.read_csv(args.csv) if args.mode == "sites" else None

    from functools import reduce

    pivots = []
    for idx, mp in enumerate(model_paths, 1):
        print(f"\n========== {mp.name} ==========")
        model = Model()
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(mp, map_location=device))
        model.eval()

        if args.mode == "full":
            result_df = run_full_length(model, args.seq, args.pdb, device)
        else:
            result_df = run_selected_sites(model, args.seq, input_df, device)

        result_df = result_df.rename(columns={"pred": f"pred{idx}", "class": f"class{idx}"})
        pivots.append(result_df[["id", "wt", "mut", f"pred{idx}", f"class{idx}"]])

    combined = reduce(lambda l, r: l.merge(r, on=["id", "wt", "mut"], how="outer"), pivots)
    combined.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}  ({len(combined)} rows, {len(model_paths)} models)")
