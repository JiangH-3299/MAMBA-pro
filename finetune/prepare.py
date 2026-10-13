import argparse
import pandas as pd


def merge_label_teacher(label_path, teacher_path):
    """Merge experimental labels with teacher model predictions (teacher column)."""
    data1 = pd.read_csv(label_path)
    data2 = pd.read_csv(teacher_path)
    data1['teacher'] = data2['fitness'].to_list()
    data1.to_csv(label_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge experimental labels with teacher predictions')
    parser.add_argument('--label', required=True, help='Path to CSV with experimental results (will be updated in-place)')
    parser.add_argument('--teacher', required=True, help='Path to CSV with teacher model predictions')
    args = parser.parse_args()

    merge_label_teacher(args.label, args.teacher)
    print(f'Merged teacher predictions into {args.label}')
