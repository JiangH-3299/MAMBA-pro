import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from model import MAMBA_pro as Model
from loader import ProDataset
from finetune.trainer import evaluate

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def test_models(test_df_path, wt_seq, batch_size, alpha, device, in_model_path, out_result_path):
    """Evaluate all model checkpoints in in_model_path on the test set."""
    test_df = pd.read_csv(test_df_path)
    test_loader = DataLoader(ProDataset(test_df, wt_seq), batch_size=batch_size, shuffle=False)

    results = []
    for model_name in sorted(os.listdir(in_model_path)):
        model_path = os.path.join(in_model_path, model_name)
        print(f'\nTesting: {model_name}')

        model = Model().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        test_loss, test_true, test_pred, test_id, test_wt, test_mut = evaluate(model, test_loader, alpha, device)
        test_spr = spearmanr(test_true, test_pred).statistic
        print(f'  Loss: {test_loss:.3f}  Spearman: {test_spr:.3f}')

        pd.DataFrame({
            'id': test_id, 'wt': test_wt, 'mut': test_mut,
            'y_true': (30.0 * np.array(test_true) - 15.0).tolist(),
            'y_pred': (30.0 * np.array(test_pred) - 15.0).tolist(),
        }).to_csv(os.path.join(out_result_path, f'{model_name}_test_detail.csv'), index=False)

        results.append({'model': model_name, 'loss': test_loss, 'spearman': test_spr})

    pd.DataFrame(results).set_index('model').to_csv(os.path.join(out_result_path, 'test_result.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test finetuned models')
    parser.add_argument('--test_dataframe', required=True)
    parser.add_argument('--wt_seq', required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--in_model', required=True)
    parser.add_argument('--out_result', required=True)
    args = parser.parse_args()

    test_models(args.test_dataframe, args.wt_seq, args.batch_size, args.alpha,
                args.device, args.in_model, args.out_result)
