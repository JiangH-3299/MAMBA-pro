import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from model import MAMBA_pro as Model
from loader import ProDataset
from finetune.trainer import train_step1, evaluate

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def pairwise_ranking_accuracy(y_pred, y_true):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    N = y_true.shape[0]
    correct = 0
    total = 0
    for i in range(N):
        for j in range(i + 1, N):
            true_diff = y_true[i] - y_true[j]
            pred_diff = y_pred[i] - y_pred[j]
            if true_diff * pred_diff > 0:
                correct += 1
            elif true_diff == 0 and pred_diff == 0:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def save_details(path_prefix, fold, name, wt, mut, y_true, y_pred):
    """Save training/validation detail CSV with scaled values."""
    df = pd.DataFrame({
        'id': name, 'wt': wt, 'mut': mut,
        'y_true': list(30.0 * np.array(y_true) - 15.0),
        'y_pred': list(30.0 * np.array(y_pred) - 15.0),
    })
    df.sort_values(by=['y_true'], inplace=True)
    df.to_csv(f'{path_prefix}/Fold{fold}_{detail}.csv', index=False)


def train(pretrained_state, model, train_df, valid_df, device, wt_seq,
          learning_rate, batch_size, alpha, keep_mlp,
          fold, out_model, out_result):

    # Load pretrained weights
    for key, prefix, module in [
        ('gcn', 4, model.gcn),
        ('graphormer', 11, model.graphormer),
        ('GRP_linear', 11, model.GRP_linear),
    ]:
        sub = {k.split('.', 1)[1]: v for k, v in pretrained_state.items() if k.startswith(key)}
        module.load_state_dict(sub)
        module.to(device)

    if keep_mlp:
        for key in ('fc1', 'fc2'):
            sub = {k.split('.', 1)[1]: v for k, v in pretrained_state.items() if k.startswith(key)}
            getattr(model, key).load_state_dict(sub)
            getattr(model, key).to(device)

    train_loader = DataLoader(ProDataset(train_df, wt_seq), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(ProDataset(valid_df, wt_seq), batch_size=batch_size, shuffle=True)

    best_acc = 0.0
    best_loss = float('inf')
    best_epoch = 0
    patience = 0
    train_losses, train_sprs = [], []
    valid_losses, valid_sprs = [], []

    print(f'max epoch: 200  sigma: 100.0  learning_rate: {learning_rate}  alpha: {alpha}')

    for epoch in range(200):
        model.train()
        train_loss = train_step1(model, train_loader, learning_rate, alpha, device)

        # Evaluate train
        _, train_true, train_pred, train_name, train_wt, train_mut = evaluate(model, train_loader, alpha, device)
        train_spr = spearmanr(train_true, train_pred).statistic
        train_acc = pairwise_ranking_accuracy(torch.tensor(train_pred), torch.tensor(train_true))
        print(f'Epoch {epoch+1:3d} | Train loss: {train_loss:.3f}  SPR: {train_spr:.3f}  Acc: {train_acc:.3f}')
        train_losses.append(train_loss)
        train_sprs.append(train_spr)

        # Evaluate validation
        valid_loss, valid_true, valid_pred, valid_name, valid_wt, valid_mut = evaluate(model, valid_loader, alpha, device)
        valid_spr = spearmanr(valid_true, valid_pred).statistic
        valid_acc = pairwise_ranking_accuracy(torch.tensor(valid_pred), torch.tensor(valid_true))
        print(f'Epoch {epoch+1:3d} | Valid loss: {valid_loss:.3f}  SPR: {valid_spr:.3f}  Acc: {valid_acc:.3f}')
        valid_losses.append(valid_loss)
        valid_sprs.append(valid_spr)

        # Save best model
        if valid_acc > best_acc or (valid_acc == best_acc and valid_loss < best_loss):
            patience = 0
            best_acc = valid_acc
            best_loss = valid_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'{out_model}/Fold{fold}_pred.pkl')

            pd.DataFrame({'id': train_name, 'wt': train_wt, 'mut': train_mut,
                          'y_true': (30.0 * np.array(train_true) - 15.0).tolist(),
                          'y_pred': (30.0 * np.array(train_pred) - 15.0).tolist(),
                          }).sort_values('y_true').to_csv(f'{out_result}/Fold{fold}_train_detail.csv', index=False)

            pd.DataFrame({'id': valid_name, 'wt': valid_wt, 'mut': valid_mut,
                          'y_true': (30.0 * np.array(valid_true) - 15.0).tolist(),
                          'y_pred': (30.0 * np.array(valid_pred) - 15.0).tolist(),
                          }).sort_values('y_true').to_csv(f'{out_result}/Fold{fold}_valid_detail.csv', index=False)

            print(f'  Model saved.  SPR={valid_spr:.3f}  Acc={valid_acc:.3f}')
        else:
            patience += 1

        if patience >= 10:
            break

    pd.DataFrame({
        'Train Loss': train_losses, 'Train_spearman': train_sprs,
        'Valid Loss': valid_losses, 'Valid_spearman': valid_sprs,
        'Best_epoch': [best_epoch] * len(train_losses),
    }).to_csv(f'{out_result}/Fold{fold}_result.csv', index=False)

    print(f'Fold {fold} best epoch: {best_epoch}')


def cross_validation(pretrained_path, data_path, device, wt_seq,
                     learning_rate, batch_size, alpha, keep_mlp,
                     fold_number, out_model, out_result):

    pretrained_state = torch.load(pretrained_path, map_location=device)
    all_df = pd.read_csv(data_path)

    kfold = KFold(n_splits=fold_number, shuffle=True)
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(all_df['mut'].values, all_df['label'].values), 1):
        print(f'\n========== Fold {fold} ==========')
        train_df = all_df.iloc[train_idx]
        valid_df = all_df.iloc[valid_idx]
        print(f'Training on {len(train_df)} examples, Validation on {len(valid_df)} examples')

        model = Model().to(device)
        train(pretrained_state, model, train_df, valid_df, device, wt_seq,
              learning_rate, batch_size, alpha, keep_mlp,
              fold, out_model, out_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Step 1 — freeze backbone, train MLP')
    parser.add_argument('--pretrained_model', required=True)
    parser.add_argument('--all_dataframe', required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--wt_seq', required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--keep_mlp', action='store_true')
    parser.add_argument('--fold_number', type=int, default=3)
    parser.add_argument('--out_model', required=True)
    parser.add_argument('--out_result', required=True)
    args = parser.parse_args()

    cross_validation(
        args.pretrained_model, args.all_dataframe, args.device, args.wt_seq,
        args.learning_rate, args.batch_size, args.alpha, args.keep_mlp,
        args.fold_number, args.out_model, args.out_result,
    )
