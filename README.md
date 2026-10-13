# MAMBA-pro

MAMBA-pro 是用于蛋白质单点突变预测的项目，提供零样本适应度评分、两阶段微调、二分类筛选器，以及 ΔΔG、ΔTm 和溶解度变化预测入口。

本目录是原 GLRP 项目的独立改名副本。展示名称为 **MAMBA-pro**，Python 类、脚本和权重文件使用 **MAMBA_pro**，因为 Python 标识符不能包含连字符。此次改名保留原有 ESM-2、GCN、图注意力和预测头结构，未引入 Mamba 状态空间模块。

## 目录

```text
MAMBA-pro/
├── MAMBA_pro_zeroshot.py        # 零样本预测
├── MAMBA_pro_predict.py         # 指定序列和权重的预测
├── MAMBA_pro_finetune.py        # 两阶段微调流水线
├── MAMBA_pro_ddg_predict.py     # ΔΔG 预测
├── MAMBA_pro_dtm_predict.py     # ΔTm 预测
├── MAMBA_pro_dsol_predict.py    # 溶解度变化三分类
├── model.py                    # MAMBA_pro 模型类
├── loader.py                   # 序列、结构特征及数据集
├── models/                     # 四个预训练权重
├── PDBs/                       # 放置用户提供的 PDB
├── finetune/                   # 微调和评估
├── filters/                    # 二分类筛选器训练和预测
├── utils/                      # ESM-2、图处理、网络及损失函数
├── tools/generate_mutations.py # 生成突变 CSV
├── requirements.txt
└── migration_manifest.json     # 原始文件 SHA-256，用于核对未修改原项目
```

## 安装

建议使用独立的 Python 3.10 或更新版本环境。在本目录执行：

```bash
conda create -n mamba-pro python=3.10 -y
conda activate mamba-pro
cd MAMBA-pro
python -m pip install -r requirements.txt
```

如果终端已经位于本目录，跳过 `cd MAMBA-pro`。GPU 运行需要与本机 CUDA 环境匹配的 PyTorch。依赖文件未锁定版本；本次基础检查使用 Python 3.12、PyTorch 2.7.1 CPU 环境。

ESM-2 使用 `esm2_t33_650M_UR50D`，产生 1280 维残基特征。首次真正提取特征时，`fair-esm` 会按其缓存机制加载或下载预训练模型，需要网络或预先准备的缓存。ESM-2 默认在双 GPU 环境使用 `cuda:1`，单 GPU 使用 `cuda:0`，无 GPU 使用 CPU；预测脚本的 `--device` 控制预测网络所在设备。CPU 提取大型模型特征可能较慢。

## 输入数据

将结构文件放在 `PDBs/<id>.pdb`。突变格式为 `A10V`：野生型 A、序列第 10 位、替换为 V；位置从 1 开始，对应提取出的序列顺序，不能直接使用存在缺号的 PDB 残基编号。`wt` 表示野生型。当前特征流程面向单点突变。

建议使用清理过的单模型 PDB，每个残基保留一个 CA 原子，序列与结构顺序一致。局部结构使用突变位点及其 32 个近邻；全局特征公式含 `长度 - 33`，因此要求结构含 **至少 34 个残基**。

指定突变的 CSV：

```csv
id,wt,mut
1abc,wt,A10V
1abc,wt,G25D
```

此处及下文的 `1abc`、突变和序列都是格式示例，使用前应替换成自己的真实数据。基础预测同一次调用中的 CSV 应对应指定的同一个 PDB 和野生型序列。

微调 CSV 需要 `id,wt,mut,label,wt_seq`；同一次基础微调应使用同一条野生型序列。筛选器训练使用相同列，`label` 必须为 0 或 1。

## 零样本预测

自动从 PDB 提取野生型序列。`full` 扫描全部位点，每个位点生成 19 个非野生型替换；`sites` 读取指定突变。

```bash
python MAMBA_pro_zeroshot.py --device cuda:0 --pdb 1abc --output zeroshot_full.csv full
python MAMBA_pro_zeroshot.py --device cuda:0 --pdb 1abc --output zeroshot_sites.csv sites --csv sites.csv
```

默认权重为 `models/MAMBA_pro.pkl`；可通过 `--model` 指定其他权重。输出列为 `id,wt,mut,fitness`，基础模型输出 sigmoid 评分，范围为 0–1。

## 指定序列预测

`--seq` 必须是完整野生型序列，长度需与提取出的 PDB 序列一致。下面的 `YOUR_FULL_WT_SEQUENCE` 为占位符。

```bash
python MAMBA_pro_predict.py --device cuda:0 --model models/MAMBA_pro.pkl --pdb 1abc --seq YOUR_FULL_WT_SEQUENCE --output predictions.csv sites --csv sites.csv
python MAMBA_pro_predict.py --device cuda:0 --model models/MAMBA_pro.pkl --pdb 1abc --seq YOUR_FULL_WT_SEQUENCE --output saturation.csv full
```

## 微调

```bash
python MAMBA_pro_finetune.py --device cuda:0 --model models/MAMBA_pro.pkl --data experimental.csv --fold_number 3 --output_path outputs/finetune
```

流水线先生成教师评分，然后冻结 GCN、图注意力和全局特征投影层进行第一阶段训练，第二阶段解冻训练。结果保存在 `model_step1`、`result_step1`、`model_step2`、`result_step2` 中，包括每折权重和预测详情。

原有训练代码使用 `(label + 15) / 30` 的数值变换，请确认实验标签符合任务的标度。流水线会向输入 CSV 写入 `teacher` 列，运行前建议使用数据副本。当前流水线在同一输入数据上评估并选择模型，不能将该结果作为独立测试集性能；正式性能评估需要另设保留测试集。

## 任务专用预测

全局选项（如 `--model`、`--device`、`--output`、`--pdb-dir`）放在子命令 `single_seq` 或 `multi_seqs` **之前**。`--chain` 指定的链同时用于序列和结构特征提取。

单条蛋白质指定链的全位点扫描：

```bash
python MAMBA_pro_ddg_predict.py --device cuda:0 --model models/MAMBA_pro_ddg.pkl --output ddg.csv --pdb-dir PDBs single_seq --pdb 1abc --chain A
python MAMBA_pro_dtm_predict.py --device cuda:0 --model models/MAMBA_pro_dtm.pkl --output dtm.csv --pdb-dir PDBs single_seq --pdb 1abc --chain A
python MAMBA_pro_dsol_predict.py --device cuda:0 --model models/MAMBA_pro_dsol.pkl --output dsol.csv --pdb-dir PDBs single_seq --pdb 1abc --chain A
```

多序列输入 CSV 需要 `id,chain,wt,mut,wt_seq`，每行的完整序列必须与对应结构链一致：

```bash
python MAMBA_pro_ddg_predict.py --device cuda:0 --model models/MAMBA_pro_ddg.pkl --output ddg.csv multi_seqs --csv task_mutations.csv
python MAMBA_pro_dtm_predict.py --device cuda:0 --model models/MAMBA_pro_dtm.pkl --output dtm.csv multi_seqs --csv task_mutations.csv
python MAMBA_pro_dsol_predict.py --device cuda:0 --model models/MAMBA_pro_dsol.pkl --output dsol.csv multi_seqs --csv task_mutations.csv
```

ΔΔG 和 ΔTm 脚本将网络输出映射为 `30 × fitness - 15`，写入 `fitness` 列。具体物理单位与正负方向取决于原权重训练标签，本代码没有提供可独立核实的训练数据说明。溶解度输出 `negative`、`neutral`、`positive` 三类，并包含 `class_id`、`confidence`、`prob_neg`、`prob_neu`、`prob_pos`。

## 二分类筛选器

训练多个随机种子的筛选器：

```bash
python filters/quick_start_batch.py --data binary_labels.csv --model models/MAMBA_pro.pkl --output outputs/filters --device cuda:0 --start_seed 42 --end_seed 51
```

使用训练后的筛选器集合。`--saved_model` 应指向实际存放筛选器 `.pkl` 文件的目录：

```bash
python filters/run_filters.py --saved_model outputs/filters/saved_model --pdb 1abc --seq YOUR_FULL_WT_SEQUENCE --output filter_predictions.csv --device cuda:0 --mode sites --csv sites.csv
```

将示例中的模型目录替换为训练产生的实际目录。结果分析入口为 `filters/analyze_batch_results.py`，完整选项可通过 `--help` 查看。

## 生成突变列表

创建 `residues.txt`，每行写一个真实位点，例如 `A10`、`G25`，再运行：

```bash
python tools/generate_mutations.py residues.txt --id 1abc --output sites.csv
```

每个位点生成 19 个替换。该工具保留原有交互行为，结束时会等待回车。

## 改名兼容性与验证

- `from model import GLRP` 改为 `from model import MAMBA_pro`；新副本入口使用 `MAMBA_pro_*.py`。
- 四个权重文件只改文件名，内容与原文件逐字节一致；网络参数名保持不变，原有 `state_dict` 可由新类加载。
- 修正零样本及筛选器的默认权重路径；ESM-2 延迟加载并支持 CPU 回退。
- 任务专用预测将 `--pdb-dir` 与链选择传递到结构特征提取；增加过短结构和越界位置检查。
- 已检查全部 Python 文件语法、六个主入口的 `--help`、四个权重的严格加载及 CPU 网络前向计算，并核对原项目文件未被修改。
- 相同权重和输入下，原模型与新模型的输出逐值一致；合成双链 PDB 检查覆盖了自定义结构目录、链筛选和三个任务入口的参数传递（ESM 特征使用模拟数据）。

本次未运行真实 ESM-2 特征提取、GPU 推理或完整微调；运行环境尚未安装 `fair-esm`，项目也未附带示例 PDB 和实验 CSV。以上检查确认改名与网络权重兼容，不能替代真实数据上的预测和训练验证。
