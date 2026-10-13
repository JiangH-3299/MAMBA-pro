import os
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from model import MAMBA_pro
from MAMBA_pro_predict import run_selected_sites
from finetune.step1 import cross_validation as step1_cv
from finetune.step2 import cross_validation as step2_cv
from finetune.tester import test_models


def run_pipeline(args):
    print('\n========== START PIPELINE ==========\n')
    os.makedirs(args.output_path, exist_ok=True)

    data = pd.read_csv(args.data)
    wt_seq = data['wt_seq'].iloc[0]

    # [1/6] zeroshot prediction as teacher (in-memory, no disk output)
    print('[1/6] Running zeroshot predictions as teacher ...')
    model = MAMBA_pro().to(args.device)
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    teacher_df = run_selected_sites(model, wt_seq, data, args.device)
    data['teacher'] = 30.0 * teacher_df['fitness'].values - 15.0
    data.to_csv(args.data, index=False)
    del model

    # [2/6] merge experimental labels with teacher predictions
    print('[2/6] Merging teacher predictions into labels ...')
    data = pd.read_csv(args.data)
    print(f'  Data shape: {data.shape}')

    # Determine learning rate & alpha from initial Spearman
    spr = spearmanr(data['label'], data['teacher']).statistic
    out_model_s1 = os.path.join(args.output_path, 'model_step1')
    out_result_s1 = os.path.join(args.output_path, 'result_step1')
    out_model_s2 = os.path.join(args.output_path, 'model_step2')
    out_result_s2 = os.path.join(args.output_path, 'result_step2')

    batch_size = len(data) // 3 + 1

    if spr >= 0.3:
        lr, alpha, keep_mlp = 0.0001, 0.7, True
        print(f'  Initial Spearman r={spr:.3f} >= 0.3 -> lr={lr}, alpha={alpha}, keep_mlp=True')
    else:
        lr, alpha, keep_mlp = 0.0005, 1.0, False
        print(f'  Initial Spearman r={spr:.3f} < 0.3 -> lr={lr}, alpha={alpha}, keep_mlp=False')

    # [3/6] Finetune Step 1
    print('\n[3/6] Finetune Step 1 (freeze backbone, train MLP) ...')
    os.makedirs(out_model_s1, exist_ok=True)
    os.makedirs(out_result_s1, exist_ok=True)
    step1_cv(
        args.model, args.data, args.device, wt_seq,
        lr, batch_size, alpha, keep_mlp, args.fold_number,
        out_model_s1, out_result_s1,
    )

    # [4/6] Test Step 1 & select best model
    print('\n[4/6] Testing Step 1 models ...')
    test_models(args.data, wt_seq, batch_size, alpha, args.device,
                out_model_s1, out_result_s1)
    s1_results = pd.read_csv(os.path.join(out_result_s1, 'test_result.csv'))
    s1_best = s1_results.sort_values('spearman', ascending=False).iloc[0]
    s1_best_path = os.path.join(out_model_s1, s1_best['model'])
    print(f'  Best Step 1 model: {s1_best_path}')

    # [5/6] Finetune Step 2
    print('\n[5/6] Finetune Step 2 (unfreeze all, full finetune) ...')
    lr2 = lr / 10.0
    os.makedirs(out_model_s2, exist_ok=True)
    os.makedirs(out_result_s2, exist_ok=True)
    step2_cv(
        s1_best_path, args.data, args.device, wt_seq,
        lr2, batch_size, alpha, args.fold_number,
        out_model_s2, out_result_s2,
    )

    # [6/6] Test Step 2
    print('\n[6/6] Testing Step 2 models ...')
    test_models(args.data, wt_seq, batch_size, alpha, args.device,
                out_model_s2, out_result_s2)
    s2_results = pd.read_csv(os.path.join(out_result_s2, 'test_result.csv'))
    s2_best = s2_results.sort_values('spearman', ascending=False).iloc[0]
    s2_best_path = os.path.join(out_model_s2, s2_best['model'])
    print(f'  Best Step 2 model: {s2_best_path}')

    print('\n========== FINISH PIPELINE ==========')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full finetuning pipeline')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fold_number', type=int, default=3)
    parser.add_argument('--output_path', required=True)

    parser.add_argument('--model', required=True, help='Pretrained model for zeroshot teacher')
    parser.add_argument('--data', required=True, help='CSV with id,wt,mut,label,wt_seq (experimental results)')
    args = parser.parse_args()

    run_pipeline(args)
