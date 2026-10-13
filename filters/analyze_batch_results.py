"""
analyze_batch_results.py ? Compare results across multiple seeds.
"""
import json, sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def analyze_batch_results(batch_results_dir=None):
    """Load batch_summary.json, aggregate seed results, generate report + plots."""
    if batch_results_dir is None:
        parent = Path(__file__).parent
        dirs = sorted(parent.glob('batch_results_*'), key=lambda d: d.stat().st_mtime, reverse=True)
        if not dirs:
            print('No batch_results_* directory found.'); return None
        batch_results_dir = dirs[0]

    batch_dir = Path(batch_results_dir)
    summary_file = batch_dir / 'batch_summary.json'
    if not summary_file.exists():
        print(f'Missing {summary_file}'); return None

    with open(summary_file) as f:
        batch = json.load(f)

    config, summary, results = batch['config'], batch['summary'], batch['results']
    print(f'Total seeds: {summary["total"]}  Success: {summary["successful"]}  Failed: {summary["failed"]}  Skipped: {summary["skipped"]}')

    # Load per-seed detailed results
    seed_accs = {'step1': {}, 'step2': {}}
    rows = []
    for seed_str, res in sorted(results.items()):
        seed = int(seed_str)
        if res['status'] != 'success':
            rows.append({'Seed': seed, 'Status': res['status']})
            continue

        seed_dirs = sorted(Path(__file__).parent.parent.glob(f'results_seed{seed}_*'))
        if not seed_dirs:
            rows.append({'Seed': seed, 'Status': 'success (no detail)'})
            continue

        sfile = seed_dirs[0] / 'summary.json'
        if sfile.exists():
            with open(sfile) as f:
                s = json.load(f)
            a1 = s['step1']['best_test_accuracy']
            a2 = s['step2']['best_test_accuracy']
            seed_accs['step1'][seed] = a1
            seed_accs['step2'][seed] = a2
            rows.append({'Seed': seed, 'Status': 'success', 'Step1 Acc': f'{a1:.4f}', 'Step2 Acc': f'{a2:.4f}'})
        else:
            rows.append({'Seed': seed, 'Status': 'success (no summary.json)'})

    print(pd.DataFrame(rows).to_string(index=False))

    if seed_accs['step2']:
        vals = list(seed_accs['step2'].values())
        print(f'\nStep2 stats:  min={min(vals):.4f}  max={max(vals):.4f}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  median={np.median(vals):.4f}')

    # Generate plots
    if plt and seed_accs['step2'] and seed_accs['step1']:
        plot_dir = batch_dir / 'analysis_plots'
        plot_dir.mkdir(exist_ok=True)
        seeds = sorted(seed_accs['step2'].keys())
        s1 = [seed_accs['step1'][s] for s in seeds]
        s2 = [seed_accs['step2'][s] for s in seeds]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        colors = ['green' if v > np.mean(s2) else 'blue' for v in s2]
        ax1.bar([str(s) for s in seeds], s2, color=colors, alpha=0.7)
        ax1.axhline(np.mean(s2), color='red', ls='--', label=f'mean={np.mean(s2):.4f}')
        ax1.set(xlabel='Seed', ylabel='Test Accuracy', title='Step 2 Test Accuracy')
        ax1.legend(); ax1.grid(axis='y', alpha=0.3)

        x = np.arange(len(seeds))
        w = 0.35
        ax2.bar(x - w/2, s1, w, label='Step 1', alpha=0.8)
        ax2.bar(x + w/2, s2, w, label='Step 2', alpha=0.8)
        ax2.set(xlabel='Seed', ylabel='Test Accuracy', title='Step 1 vs Step 2')
        ax2.set_xticks(x); ax2.set_xticklabels([str(s) for s in seeds])
        ax2.legend(); ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_dir / 'seed_comparison.png', dpi=150)
        plt.close()
        print(f'Plots saved to {plot_dir}')

    return {'batch_dir': batch_dir, 'summary': summary, 'seed_accuracies': seed_accs}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze batch training results')
    parser.add_argument('--batch_dir', default=None, help='Path to batch_results_* directory (default: latest)')
    args = parser.parse_args()

    path = args.batch_dir
    result = analyze_batch_results(path)
    if result:
        print(f'\nReport: {result["batch_dir"]}')
