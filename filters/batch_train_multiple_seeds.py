"""
batch_train_multiple_seeds.py -- Run auto_train across multiple seeds.
"""

import os, sys, json, traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_single_seed(data_path, train_ratio, lr_step1, lr_step2, fold_number, seed, pretrained_path=None, output_dir=None, device=None):
    """Train a single seed via auto_train.main()."""
    print(f"\n{'='*80}\n{'SEED ' + str(seed):^80}\n{'='*80}")
    try:
        from filters.auto_train import main
        main(data_path, train_ratio, lr_step1, lr_step2, fold_number, seed=seed,
             pretrained_path=pretrained_path, output_dir=output_dir, device=device)
        print(f"\nSeed {seed} completed.")
        return True, None
    except Exception as e:
        print(f"\nSeed {seed} failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, str(e)


def batch_train_multiple_seeds(data_path, train_ratio, lr_step1, lr_step2, fold_number,
                               start_seed, end_seed, skip_seeds=None, pretrained_path=None, output_dir=None, device=None):
    """Run auto_train for every seed in [start_seed, end_seed], skipping skip_seeds."""
    skip_seeds = skip_seeds or []
    output_dir = Path(output_dir) if output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = output_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir.mkdir(exist_ok=True)
    log_file = batch_dir / 'batch_training_log.txt'

    summary = {
        'config': dict(data_path=str(data_path), train_ratio=train_ratio, lr_step1=lr_step1,
                       lr_step2=lr_step2, fold_number=fold_number, start_seed=start_seed,
                       end_seed=end_seed, skip_seeds=skip_seeds, output_dir=str(output_dir),
                       batch_start_time=datetime.now().isoformat()),
        'results': {}
    }
    successful, failed = [], {}

    for seed in range(start_seed, end_seed + 1):
        if seed in skip_seeds:
            summary['results'][seed] = {'status': 'skipped'}
            continue
        ok, err = run_single_seed(data_path, train_ratio, lr_step1, lr_step2, fold_number, seed,
                                  pretrained_path=pretrained_path, output_dir=str(output_dir), device=device)
        if ok:
            successful.append(seed)
            summary['results'][seed] = {'status': 'success'}
        else:
            failed[seed] = err
            summary['results'][seed] = {'status': 'failed', 'error': err}
            with open(log_file, 'a') as f:
                f.write(f"\nSEED {seed} FAILED\n  {err}\n{'='*60}\n")

    total = end_seed - start_seed + 1 - len(skip_seeds)
    print(f'\n{"="*80}')
    print(f'Batch complete: {len(successful)}/{total} succeeded, {len(failed)} failed, {len(skip_seeds)} skipped')
    if failed:
        print(f'Failed seeds: {list(failed.keys())}')

    summary['batch_end_time'] = datetime.now().isoformat()
    summary['summary'] = dict(total=total, successful=len(successful), failed=len(failed),
                              skipped=len(skip_seeds), successful_seeds=successful,
                              failed_seeds=list(failed.keys()))
    json.dump(summary, open(batch_dir / 'batch_summary.json', 'w'), indent=4)
    print(f'Summary saved: {batch_dir / "batch_summary.json"}')

    # Select best model across all seeds
    print(f'\nSelecting best model across seeds...')
    from filters.auto_train import select_best_model
    select_best_model(batch_dir)

    return batch_dir, summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Batch training across multiple seeds')
    parser.add_argument('--data', required=True, help='CSV with id,wt,mut,label,wt_seq')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--lr_step1', type=float, default=0.001)
    parser.add_argument('--lr_step2', type=float, default=0.0001)
    parser.add_argument('--fold_number', type=int, default=5)
    parser.add_argument('--start_seed', type=int, default=42)
    parser.add_argument('--end_seed', type=int, default=51)
    parser.add_argument('--skip_seeds', type=int, nargs='*', default=[], help='Seeds to skip')
    parser.add_argument('--model', '-m', default=None, help='Pretrained .pkl path (default: models/MAMBA_pro.pkl)')
    parser.add_argument('--output', '-o', default=None, help='Output directory (default: filters/)')
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f'ERROR: {args.data} not found'); sys.exit(1)
    batch_dir, summary = batch_train_multiple_seeds(
        data_path=args.data, train_ratio=args.train_ratio, lr_step1=args.lr_step1,
        lr_step2=args.lr_step2, fold_number=args.fold_number,
        start_seed=args.start_seed, end_seed=args.end_seed, skip_seeds=args.skip_seeds,
        pretrained_path=args.model, output_dir=args.output)
    print(f'All results: {batch_dir}')
