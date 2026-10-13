# MAMBA-pro

## Overview

Protein engineering is fundamentally limited by the vastness of protein sequence space and the scarcity of experimentally characterized variants. **MAMBA-pro combines mutation-centered global and local representations to support protein mutation scoring and adaptation to experimental objectives.** Its pretrained representations can be fine-tuned using experimental mutation data.

The framework integrates ESM-2 sequence embeddings with graph convolutional networks (GCN), graph attention, and multi-head attention to represent changes between wild-type and mutant proteins. It provides zero-shot scoring, two-stage fine-tuning, full-sequence scanning, selected-mutation prediction, binary filters, and task-specific prediction for ΔΔG, ΔSol, and ΔTm.

<img width="6286" height="3674" alt="图片5" src="https://github.com/user-attachments/assets/c6b32184-dcd0-4b0e-8e83-6781e0bffe25" />

## Repository Structure

```text
MAMBA-pro/
├── MAMBA_pro_ddg_predict.py   # ΔΔG prediction
├── MAMBA_pro_dsol_predict.py  # Solubility-change classification
├── MAMBA_pro_dtm_predict.py   # ΔTm prediction
├── MAMBA_pro_predict.py       # Prediction with a user-provided sequence
├── MAMBA_pro_zeroshot.py      # Zero-shot prediction
├── MAMBA_pro_finetune.py      # Two-stage fine-tuning pipeline
├── loader.py                 # Feature calculation and data loading
├── model.py                  # MAMBA_pro model
├── requirements.txt          # Python dependencies
├── PDBs/                     # User-provided protein structures
├── filters/                  # Binary filter training and inference
├── finetune/                 # Fine-tuning modules
├── models/                   # Pretrained models
│   ├── MAMBA_pro.pkl
│   ├── MAMBA_pro_ddg.pkl
│   ├── MAMBA_pro_dsol.pkl
│   └── MAMBA_pro_dtm.pkl
├── utils/                    # ESM-2, graph, and attention components
└── tools/
    └── generate_mutations.py # Mutation-list generation
```

## Installation

Run the following commands from the repository root:

```bash
conda create -n mamba-pro python=3.10 -y
conda activate mamba-pro
python -m pip install -r requirements.txt
```

For GPU execution, install a CUDA-compatible PyTorch build for your system. The examples below use `--device cuda:0`; select another available GPU or use `--device cpu` as needed.

ESM-2 (`esm2_t33_650M_UR50D`) is loaded on the first feature-extraction call and requires downloaded weights or an existing local cache. Its embedding device is selected automatically: `cuda:1` when at least two GPUs are available, otherwise `cuda:0` or CPU. The command-line `--device` option controls the prediction network.

## Usage

**Place your structures in `PDBs/` under the repository root**, for example `PDBs/1abc.pdb`. Use the filename without `.pdb` as the protein `id`.

Run `python <script> --help` for the available options. Replace `1abc`, example mutations, and `YOUR_FULL_WT_SEQUENCE` with your own data. Sequence arguments must contain the complete wild-type sequence.

Mutations use the format `A10V`: alanine at sequence position 10 is replaced by valine. Positions are **1-based sequence indices**, which may differ from PDB residue numbers. Use structures with consistent sequence and CA-atom ordering and at least 34 residues in the structure used for feature extraction.

### 1. Zero-shot prediction

Predict directly from a PDB file; the wild-type sequence is extracted automatically.

**Full scan** — generate all 19 non-wild-type substitutions at each position:

```bash
python MAMBA_pro_zeroshot.py --device cuda:0 -m models/MAMBA_pro.pkl -p 1abc -o MAMBA_pro_pred_zeroshot.csv full
```

**Selected mutations:**

```bash
python MAMBA_pro_zeroshot.py --device cuda:0 -m models/MAMBA_pro.pkl -p 1abc -o MAMBA_pro_pred_zeroshot_sites.csv sites --csv sites.csv
```

`sites.csv` requires the columns `id,wt,mut`:

```csv
id,wt,mut
1abc,wt,A10V
1abc,wt,G25D
```

Use mutations matching the selected protein's sequence. The output contains `id,wt,mut,fitness`; the base model returns a sigmoid score between 0 and 1.

### 2. Fine-tuning

Adapt the pretrained model using experimental data and teacher predictions through two training stages: first freeze the backbone and train the MLP, then fine-tune all parameters.

Input CSV columns: `id,wt,mut,label,wt_seq`. The `id` is the PDB filename without `.pdb`, `label` is the experimental measurement, and `wt_seq` is the complete wild-type sequence. Use a single wild-type sequence per fine-tuning run.

```bash
python MAMBA_pro_finetune.py --device cuda:0 --output_path round2 --model models/MAMBA_pro.pkl --data R1_exp.csv
```

Weights and prediction results are saved under `round2/model_step1`, `round2/result_step1`, `round2/model_step2`, and `round2/result_step2`.

The pipeline adds or replaces a `teacher` column in the input CSV, so use a working copy of experimental data. Training transforms labels using `(label + 15) / 30`; ensure that your label scale is appropriate. Model selection evaluates the supplied data; use a separate held-out dataset for independent performance assessment.

### 3. Prediction

Use a user-provided wild-type sequence with structural information from the PDB file. The sequence must align with the structure and match the extracted PDB sequence length. Supply pretrained or fine-tuned weights through `--model`.

**Full scan:**

```bash
python MAMBA_pro_predict.py --device cuda:0 -m models/MAMBA_pro.pkl -p 1abc -s YOUR_FULL_WT_SEQUENCE -o MAMBA_pro_pred.csv full
```

**Selected mutations:**

```bash
python MAMBA_pro_predict.py --device cuda:0 -m models/MAMBA_pro.pkl -p 1abc -s YOUR_FULL_WT_SEQUENCE -o MAMBA_pro_pred_sites.csv sites --csv sites.csv
```

### 4. Filters

Train binary classifiers with `label = 0` (negative) or `label = 1` (positive). Training CSV columns: `id,wt,mut,label,wt_seq`.

**Training filters:**

```bash
python filters/quick_start_batch.py --data full_data.csv -m models/MAMBA_pro.pkl -o results --device cuda:0
```

Each seed produces a directory such as `results/results_seed42_<timestamp>/`, containing `step1/models` and `step2/models`.

**Using filters:**

Set `--saved_model` to the actual directory containing the trained `.pkl` files. Replace `<timestamp>` below with the value generated during training.

```bash
python filters/run_filters.py --saved_model "results/results_seed42_<timestamp>/step2/models" -p 1abc -s YOUR_FULL_WT_SEQUENCE -o filter_preds.csv --device cuda:0 --mode full
```

For selected-mutation prediction, replace `--mode full` with `--mode sites --csv sites.csv`. The CSV requires `id,wt,mut`.

### 5. ΔΔG / ΔSol / ΔTm

The three task-specific scripts share the same command structure, with a corresponding model for each task.

**Single protein — saturation mutagenesis of one PDB chain:**

```bash
python MAMBA_pro_ddg_predict.py --device cuda:0 -m models/MAMBA_pro_ddg.pkl -o ddg_predictions.csv single_seq -p 1abc --chain A
python MAMBA_pro_dsol_predict.py --device cuda:0 -m models/MAMBA_pro_dsol.pkl -o dsol_predictions.csv single_seq -p 1abc --chain A
python MAMBA_pro_dtm_predict.py --device cuda:0 -m models/MAMBA_pro_dtm.pkl -o dtm_predictions.csv single_seq -p 1abc --chain A
```

**Multiple proteins — predict mutations listed in a CSV:**

```bash
python MAMBA_pro_ddg_predict.py --device cuda:0 -m models/MAMBA_pro_ddg.pkl -o ddg_predictions.csv multi_seqs --csv test.csv
python MAMBA_pro_dsol_predict.py --device cuda:0 -m models/MAMBA_pro_dsol.pkl -o dsol_predictions.csv multi_seqs --csv test.csv
python MAMBA_pro_dtm_predict.py --device cuda:0 -m models/MAMBA_pro_dtm.pkl -o dtm_predictions.csv multi_seqs --csv test.csv
```

- `single_seq`: scan one PDB chain; the default chain is `A`.
- `multi_seqs`: read a CSV with columns `id,chain,wt,mut,wt_seq`. Each sequence must match its corresponding structural chain.
- PDB files are read from `./PDBs/` by default. To change this, place `--pdb-dir PATH` before `single_seq` or `multi_seqs`.
- ΔΔG and ΔTm predictions are written to `fitness` after the transformation `30 × score − 15`. Units and sign conventions follow the training labels and are not specified by the scripts.
- ΔSol predicts three classes: `negative`, `neutral`, and `positive`, with class probabilities and confidence scores.

## Reference

Publication and citation details will be added when available.
