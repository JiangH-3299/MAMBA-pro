"""
auto_train.py - Classification-based training pipeline (BCEWithLogitsLoss)
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from sklearn.model_selection import KFold
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import multiprocessing

from loader import full_information
from utils.GA import GraphAttention
from utils.GCN import GCN
from utils.loss_function import loss

# Model hyperparameters
GCN_input_dim = 1280
GCN_hidden_dim = 256
GCN_output_dim = 63
Attention_output_dim = 64
Num_classes = 1

torch.set_default_dtype(torch.float32)
warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Dataset
# ============================================================================

class ProDataset(Dataset):
    """Dataset for classification training (label only, no teacher)."""
    def __init__(self, dataframe):
        self.Ids = dataframe['id'].values
        self.wts = dataframe['wt'].values
        self.muts = dataframe['mut'].values
        self.labels = dataframe['label'].values
        self.wt_seqs = dataframe['wt_seq'].values

    def __getitem__(self, index):
        Id = self.Ids[index]
        wt = self.wts[index]
        mut = self.muts[index]
        label = self.labels[index]
        wt_sequence = self.wt_seqs[index]

        _, _, _, LRP_wt, LRP_mut, GRP_wt, GRP_mut, contact_map, bias, centrality = \
            full_information(wt_sequence, Id, wt, mut)

        return LRP_wt, LRP_mut, GRP_wt, GRP_mut, contact_map, bias, centrality, label

    def __len__(self):
        return len(self.labels)


# ============================================================================
# Model
# ============================================================================

class Model(nn.Module):
    """Main model architecture"""
    def __init__(self, num_heads=8):
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
        padding_mask = x[:, :, 0].eq(0)
        
        x = self.gcn(x, adj)
        
        x = torch.cat((x, centrality.unsqueeze(-1)), dim=-1)
        x = x.masked_fill(padding_mask.unsqueeze(2), float(0))
        x, attn_weight = self.graphormer(x, attn_bias, padding_mask)
        
        return x, attn_weight
    
    def forward(self, x1, x2, x3, x4, adj, attn_bias, centrality):
        x1, _ = self.PSSR_module(x1, adj, attn_bias, centrality)
        x2, _ = self.PSSR_module(x2, adj, attn_bias, centrality)
        
        GRP = x4 - x3
        GRP = self.GRP_linear(GRP)
        
        x1 = torch.mean(x1, dim=1)
        x1 = self.LN1(x1)
        x2 = torch.mean(x2, dim=1)
        x2 = self.LN2(x2)
        
        LRP = x2 - x1
        x = torch.cat((LRP, GRP), dim=-1)
        
        output = self.classifier(x)

        return output.squeeze(-1)


# ============================================================================
# Training helpers
# ============================================================================

def _load(arr):
    return torch.squeeze(arr).float().to(device)

def train_step(model, loader, lr, step):
    """Train one epoch."""
    for p in model.gcn.parameters():        p.requires_grad = (step == 2)
    for p in model.graphormer.parameters(): p.requires_grad = (step == 2)
    for p in model.GRP_linear.parameters(): p.requires_grad = (step == 2)

    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0001)
    criterion = nn.BCEWithLogitsLoss()

    total_loss = correct = total = 0
    for data in tqdm(loader):
        opt.zero_grad()
        lrp_wt, lrp_mut, grp_wt, grp_mut, cmap, bias, cent, labels = data
        lrp_wt, lrp_mut, grp_wt, grp_mut = _load(lrp_wt), _load(lrp_mut), _load(grp_wt), _load(grp_mut)
        cmap, bias, cent, labels = _load(cmap), _load(bias), _load(cent), _load(labels)

        logits = model(lrp_wt, lrp_mut, grp_wt, grp_mut, cmap, bias, cent).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        correct += ((torch.sigmoid(logits) > 0.5) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


def evaluate(model, loader):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    records = {'true': [], 'pred': []}
    total_loss = correct = total = 0

    for data in tqdm(loader):
        with torch.no_grad():
            lrp_wt, lrp_mut, grp_wt, grp_mut, cmap, bias, cent, labels = data
            lrp_wt, lrp_mut, grp_wt, grp_mut = _load(lrp_wt), _load(lrp_mut), _load(grp_wt), _load(grp_mut)
            cmap, bias, cent, labels = _load(cmap), _load(bias), _load(cent), _load(labels)

            logits = model(lrp_wt, lrp_mut, grp_wt, grp_mut, cmap, bias, cent).squeeze(-1)
            total_loss += criterion(logits, labels).item()

            y_pred = torch.sigmoid(logits).float()
            correct += ((y_pred > 0.5) == labels).sum().item()
            total += labels.size(0)
            records['true'].extend(labels.cpu().tolist())
            records['pred'].extend(y_pred.cpu().tolist())

    return total_loss / max(len(loader), 1), correct / max(total, 1), records


def train_fold(pretrained, train_df, valid_df, lr, fold, step, model_dir, result_dir):
    """Single fold: init from pretrained, train, save best."""
    model = Model().to(device)
    for key in ('gcn', 'graphormer', 'GRP_linear'):
        sub = {k.split('.', 1)[1]: v for k, v in pretrained.items() if k.startswith(key)}
        getattr(model, key).load_state_dict(sub)

    train_loader = DataLoader(ProDataset(train_df), batch_size=8, shuffle=True)
    valid_loader = DataLoader(ProDataset(valid_df), batch_size=8, shuffle=True)

    best_acc = best_loss = 0
    best_epoch = patience = 0
    best_train_acc = 0

    for epoch in range(500):
        model.train()
        tr_loss, tr_acc = train_step(model, train_loader, lr, step)
        val_loss, val_acc, val_rec = evaluate(model, valid_loader)
        print(f'Epoch {epoch+1:3d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {val_loss:.4f}/{val_acc:.4f}')

        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss, best_epoch, patience, best_train_acc = val_acc, val_loss, epoch+1, 0, tr_acc
            torch.save(model.state_dict(), model_dir / f'Fold{fold}_best_model.pkl')
            pd.DataFrame(val_rec).sort_values('true').to_csv(
                result_dir / f'Fold{fold}_valid_detail.csv', index=False)
        else:
            patience += 1
            if patience >= 10:
                break

    pd.DataFrame({'Best_epoch': [best_epoch], 'Train_Acc': [best_train_acc], 'Val_Acc': [best_acc]}).to_csv(
        result_dir / f'Fold{fold}_result.csv', index=False)
    return best_epoch, best_acc, best_train_acc


def cross_validate(train_df, lr, fold_number, pretrained_path, model_dir, result_dir, step):
    pretrained = torch.load(pretrained_path, map_location=device)
    kfold = KFold(n_splits=fold_number, shuffle=True, random_state=42)
    results = []

    for fold, (tr_idx, va_idx) in enumerate(kfold.split(train_df['mut'].values, train_df['label'].values), 1):
        print(f'\n========== Fold {fold} ==========')
        best_epoch, best_acc, best_train_acc = train_fold(pretrained, train_df.iloc[tr_idx], train_df.iloc[va_idx],
                                          lr, fold, step, model_dir, result_dir)
        results.append({'fold': fold, 'best_epoch': best_epoch, 'best_acc': best_acc, 'train_acc': best_train_acc})
    return results


def test_models(test_df, model_paths, result_path, step):
    loader = DataLoader(ProDataset(test_df), batch_size=16, shuffle=False)
    all_records = []

    for idx, mp in enumerate(sorted(model_paths)):
        model = Model().to(device)
        model.load_state_dict(torch.load(mp, map_location=device))
        _, acc, rec = evaluate(model, loader)
        print(f'Model {idx+1}: test_acc={acc:.4f}')
        pd.DataFrame(rec).to_csv(result_path / f'step{step}_model{idx+1}_test.csv', index=False)
        all_records.append({'model': f'Fold{idx+1}', 'test_acc': acc, 'path': str(mp)})

    summary_df = pd.DataFrame(all_records)
    summary_df.to_csv(
        result_path / f'step{step}_test_summary.csv', index=False)
    best = summary_df.loc[summary_df['test_acc'].idxmax()]
    return best['model'], best['test_acc'], Path(best['path'])


# ============================================================================
# Main
# ============================================================================

def main(data_path, train_ratio, lr_step1, lr_step2, fold_number, seed=42, pretrained_path=None, output_dir=None, device=None):
    if device is not None:
        globals()['device'] = torch.device(device)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = Path(output_dir) if output_dir else Path(__file__).parent
    base = base_dir / f'results_seed{seed}_{ts}'
    s1_m, s1_r = base/'step1/models', base/'step1/results'
    s2_m, s2_r = base/'step2/models', base/'step2/results'
    for p in [s1_m, s1_r, s2_m, s2_r]: p.mkdir(parents=True, exist_ok=True)

    pretrained_path = Path(pretrained_path) if pretrained_path else Path(__file__).resolve().parent.parent / 'models' / 'MAMBA_pro.pkl'

    config = dict(data_path=str(data_path), train_ratio=train_ratio, lr_step1=lr_step1,
                  lr_step2=lr_step2, fold_number=fold_number, seed=seed, device=str(device), output_dir=str(base_dir))
    json.dump(config, open(base/'config.json', 'w'), indent=4)

    print(f'\n{"="*80}\nAUTO-TRAIN  seed={seed}\n{"="*80}')

    all_df = pd.read_csv(data_path)
    n_train = int(len(all_df) * train_ratio)
    idx = np.random.permutation(len(all_df))
    train_df = all_df.iloc[idx[:n_train]].reset_index(drop=True)
    test_df = all_df.iloc[idx[n_train:]].reset_index(drop=True)
    print(f'Train: {len(train_df)}  Test: {len(test_df)}')

    print('\n[Step 1] KFold (backbone frozen)...')
    s1_folds = cross_validate(train_df, lr_step1, fold_number, pretrained_path, s1_m, s1_r, step=1)
    print('\n[Step 1] Testing...')
    _, best_acc1, best_path1 = test_models(test_df, list(s1_m.glob('Fold*_best_model.pkl')), s1_r, step=1)

    print('\n[Step 2] KFold (full finetune)...')
    s2_folds = cross_validate(train_df, lr_step2, fold_number, best_path1, s2_m, s2_r, step=2)
    print('\n[Step 2] Testing...')
    _, best_acc2, _ = test_models(test_df, list(s2_m.glob('Fold*_best_model.pkl')), s2_r, step=2)

    summary = {'step1': {'fold_results': s1_folds, 'best_test_accuracy': best_acc1},
               'step2': {'fold_results': s2_folds, 'best_test_accuracy': best_acc2}}
    json.dump(summary, open(base/'summary.json', 'w'), indent=4)

    select_best_model_per_seed(base)

    print(f'\nDone. Step1 acc={best_acc1:.4f}  Step2 acc={best_acc2:.4f}')
    print(f'Results: {base}')

def select_best_model_per_seed(seed_dir):
    """Pick best model from a single seed result directory by (test, val, train) acc."""
    import shutil, re
    seed_dir = Path(seed_dir)

    # Extract seed number from directory name: results_seed42_xxx
    m = re.search(r'_seed(\d+)_', seed_dir.name)
    seed_label = m.group(1) if m else 'unknown'

    candidates = []
    for step_name in ('step1', 'step2'):
        test_csv = seed_dir / step_name / 'results' / f'{step_name}_test_summary.csv'
        if not test_csv.exists():
            continue
        with open(seed_dir / 'summary.json') as f:
            summary = json.load(f)
        fold_results = summary[step_name]['fold_results']

        for _, row in pd.read_csv(test_csv).iterrows():
            fold_idx = int(row['model'].replace('Fold', ''))
            fi = next((f for f in fold_results if f['fold'] == fold_idx), {})
            candidates.append({
                'step': step_name, 'fold': fold_idx,
                'test_acc': row['test_acc'], 'val_acc': fi.get('best_acc', 0),
                'train_acc': fi.get('train_acc', 0),
                'model_path': row['path'],
            })

    if not candidates:
        print(f'[select_best_model_per_seed] No candidates in {seed_dir.name}')
        return None

    df = pd.DataFrame(candidates).sort_values(
        ['test_acc', 'val_acc', 'train_acc'],
        ascending=[False, False, False]).reset_index(drop=True)
    best = df.iloc[0]

    selected_dir = seed_dir / 'selected_model'
    selected_dir.mkdir(exist_ok=True)
    shutil.copy2(Path(best['model_path']), selected_dir / f'best_model_seed{seed_label}.pkl')
    df.to_csv(selected_dir / 'ranking.csv', index=False)

    print(f'\n--- Per-seed best model in {seed_dir.name} ---')
    print(f'  {best["step"]} Fold{int(best["fold"])} | '
          f'test={best["test_acc"]:.4f} val={best["val_acc"]:.4f} train={best["train_acc"]:.4f}')
    print(f'  Saved to: {selected_dir / f"best_model_seed{seed_label}.pkl"}')
    return best


def select_best_model(batch_dir):
    """Scan all seed results under batch_dir, pick best model by (test, val, train) acc."""
    import shutil
    batch_dir = Path(batch_dir)
    summary_file = batch_dir / 'batch_summary.json'
    if not summary_file.exists():
        print(f'[select_best_model] No batch_summary.json in {batch_dir}')
        return None

    with open(summary_file) as f:
        batch = json.load(f)

    candidates = []
    for seed_str, info in sorted(batch['results'].items()):
        if info['status'] != 'success':
            continue
        seed = int(seed_str)

        # Find this seed's result directory
        seed_dirs = sorted(batch_dir.parent.glob(f'results_seed{seed}_*'))
        if not seed_dirs:
            continue
        seed_dir = seed_dirs[-1]  # latest
        if not (seed_dir / 'summary.json').exists():
            continue

        with open(seed_dir / 'summary.json') as f:
            summary = json.load(f)

        for step_name in ('step1', 'step2'):
            fold_results = summary[step_name]['fold_results']
            test_csv = seed_dir / step_name / 'results' / f'{step_name}_test_summary.csv'
            if not test_csv.exists():
                continue

            for _, row in pd.read_csv(test_csv).iterrows():
                fold_idx = int(row['model'].replace('Fold', ''))
                fi = next((f for f in fold_results if f['fold'] == fold_idx), {})
                candidates.append({
                    'seed': seed, 'step': step_name, 'fold': fold_idx,
                    'test_acc': row['test_acc'], 'val_acc': fi.get('best_acc', 0),
                    'train_acc': fi.get('train_acc', 0),
                    'model_path': row['path'],
                })

    if not candidates:
        print('[select_best_model] No candidates found.')
        return None

    df = pd.DataFrame(candidates).sort_values(
        ['test_acc', 'val_acc', 'train_acc'],
        ascending=[False, False, False]).reset_index(drop=True)
    best = df.iloc[0]

    selected_dir = batch_dir / 'selected_model'
    selected_dir.mkdir(exist_ok=True)
    shutil.copy2(Path(best['model_path']), selected_dir / 'best_model.pkl')
    df.to_csv(selected_dir / 'ranking.csv', index=False)

    print(f'\n{"="*60}')
    print(f'Best model selected: seed={int(best["seed"])} {best["step"]} Fold{int(best["fold"])}')
    print(f'  Test  acc: {best["test_acc"]:.4f}')
    print(f'  Val   acc: {best["val_acc"]:.4f}')
    print(f'  Train acc: {best["train_acc"]:.4f}')
    print(f'  Saved to: {selected_dir / "best_model.pkl"}')
    print(f'  Ranking:  {selected_dir / "ranking.csv"}')
    print(f'{"="*60}')
    return best


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Classification-based training pipeline (BCEWithLogitsLoss)')
    parser.add_argument('--data', required=True, help='CSV with id,wt,mut,label,wt_seq')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--lr_step1', type=float, default=0.001)
    parser.add_argument('--lr_step2', type=float, default=0.0001)
    parser.add_argument('--fold_number', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', '-m', default=None, help='Pretrained .pkl path (default: models/MAMBA_pro.pkl)')
    args = parser.parse_args()

    multiprocessing.set_start_method('spawn', force=True)
    main(args.data, train_ratio=args.train_ratio, lr_step1=args.lr_step1,
         lr_step2=args.lr_step2, fold_number=args.fold_number, seed=args.seed,
         pretrained_path=args.model)
