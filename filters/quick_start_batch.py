"""Quick start for batch training across multiple seeds."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from filters.batch_train_multiple_seeds import batch_train_multiple_seeds

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Quick-start batch training')
    parser.add_argument('--data', required=True, help='CSV with id,wt,mut,label,wt_seq')
    parser.add_argument('--model', '-m', default=None, help='Pretrained .pkl path (default: models/MAMBA_pro.pkl)')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--lr_step1', type=float, default=0.001)
    parser.add_argument('--lr_step2', type=float, default=0.0001)
    parser.add_argument('--fold_number', type=int, default=5)
    parser.add_argument('--start_seed', type=int, default=42)
    parser.add_argument('--end_seed', type=int, default=51)
    parser.add_argument('--skip_seeds', type=int, nargs='*', default=[])
    parser.add_argument('--output', '-o', default=None, help='Output directory (default: filters/)')
    parser.add_argument('--device', default=None, help='Device (e.g. cuda:0, cuda:1, cpu)')
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f'ERROR: {args.data} not found'); sys.exit(1)
    batch_dir, summary = batch_train_multiple_seeds(
        data_path=args.data, train_ratio=args.train_ratio, lr_step1=args.lr_step1,
        lr_step2=args.lr_step2, fold_number=args.fold_number,
        start_seed=args.start_seed, end_seed=args.end_seed, skip_seeds=args.skip_seeds,
        pretrained_path=args.model, output_dir=args.output, device=args.device)
    print(f'\nResults: {batch_dir}')
