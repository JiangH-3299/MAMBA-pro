import argparse
import glob
import os
import sys
import pandas as pd


def parse_residue(s):
    """Parse a residue string like 'S16' into (wt_aa, position)."""
    s = s.strip()
    wt_aa = s[0].upper()
    pos = int(s[1:])
    return wt_aa, pos


def generate_saturation_mutations(residues, protein_id):
    """Generate all single-point saturation mutations for given residues."""
    ALL_AA = 'ACDEFGHIKLMNPQRSTVWY'
    records = []

    for res in residues:
        wt_aa, pos = parse_residue(res)
        for mut_aa in (aa for aa in ALL_AA if aa != wt_aa):
            mut = f'{wt_aa}{pos}{mut_aa}'
            records.append({'id': protein_id, 'wt': 'wt', 'mut': mut})

    return pd.DataFrame(records, columns=['id', 'wt', 'mut'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate saturation mutation CSV')
    parser.add_argument('input', nargs='?', help='TXT file with one residue per line')
    parser.add_argument('--id', help='Protein/PDB identifier')
    parser.add_argument('--output', '-o', default='saturation_mutations.csv', help='Output CSV path')
    args = parser.parse_args()

    # --- Auto-detect TXT file if double-clicked (no args) ---
    if args.input is None:
        txt_files = glob.glob('*.txt')
        if not txt_files:
            print('No .txt files found in current directory.')
            input('\nPress Enter to exit...')
            sys.exit(1)
        elif len(txt_files) == 1:
            args.input = txt_files[0]
        else:
            print('Multiple .txt files found:')
            for i, f in enumerate(txt_files):
                print(f'  {i + 1}. {f}')
            choice = int(input('Choose (number): ')) - 1
            args.input = txt_files[choice]
        print(f'Using: {args.input}')

    # --- Prompt for protein ID if not provided ---
    if args.id is None:
        args.id = input('Enter protein/PDB ID: ')

    # --- Generate ---
    with open(args.input) as f:
        residues = [line.strip() for line in f if line.strip()]

    df = generate_saturation_mutations(residues, args.id)
    df.to_csv(args.output, index=False)

    print(f'\nGenerated {len(df)} mutations for {len(residues)} residues')
    print(f'Saved to {args.output}')
    input('\nPress Enter to exit...')
