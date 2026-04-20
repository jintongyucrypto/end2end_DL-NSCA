
# End-to-End Non-Profiled Side-Channel Analysis on Long Raw Traces

This repository contains the implementation of **convWIN-MCR**, the end-to-end deep learning framework proposed in the paper:

> **End-to-End Non-Profiled Side-Channel Analysis on Long Raw Traces**

---

## Overview

Side-channel analysis (SCA) typically requires preprocessing (e.g., alignment, feature selection) before applying machine learning models. This work proposes an **end-to-end non-profiled SCA framework** that operates directly on long raw power/EM traces without manual preprocessing.

The key components are:

- **convWIN**: A convolutional windowing module that applies a sliding window over the raw trace via 1D convolution and average pooling, automatically extracting local features at configurable scales (controlled by `kernel` and `stride`).
- **MCR (Multi-Channel Regression)**: A multi-channel output head that simultaneously regresses the Hamming Weight (HW) of all 256 key hypotheses in a single forward pass, enabling efficient non-profiled key recovery.

The architecture avoids the need for trace alignment or point-of-interest selection, making it directly applicable to long raw traces.

---

## Framework

The implementation is in the folder `end2end_DL_NSCA_framework/`.

### `convWIN-mcr_raw_hw_gs_paras.py`

Main training script that sweeps over hyperparameters (`stride`) and trace counts to evaluate key recovery performance.

**Model architecture (`convWIN_MCR`):**

```
Input raw trace
    └─> Conv1d(1, kernel, kernel_size=1) + BN + ReLU + AvgPool1d(stride)   [convWIN]
    └─> Flatten
    └─> Linear(input_size, 800) + BN + ReLU                                [shared layer 1]
    └─> Linear(800, 1000) + BN + ReLU                                      [shared layer 2]
    └─> Linear(1000, 1) × 256 channels                                     [MCR output head]
Output: HW predictions for all 256 key byte hypotheses
```

**Key hyperparameters:**

| Parameter | Description | Default |
|---|---|---|
| `kernel` | Number of conv filters / pool window factor | 8 |
| `stride` | Pooling stride (controls compression ratio) | 200 / 500 / 1000 / 2000 |
| `num_features` | Raw trace length | 100000 |
| `shared_layer_size1` | Hidden size of shared layer 1 | 800 |
| `shared_layer_size2` | Hidden size of shared layer 2 | 1000 |
| `num_channels` | Number of key hypotheses (fixed for AES byte) | 256 |
| `learning_rate` | Adam optimizer learning rate | 0.001 |
| `batch_size` | Training batch size | 50 |
| `epochs` | Training epochs per experiment | 50 |

**Loss function:** Per-channel MSE between predicted HW and true HW, summed over all 256 channels.

**Key rank metric:** After training, channels are sorted by final-epoch loss. The rank of the correct key byte (index 224 in the label encoding) is reported as the guessing entropy proxy (`loss_ge`).

---

## Datasets

### ASCAD (ASCADv1 fixed/random key)

Standard benchmark dataset for SCA research. Set `target = 'ASCAD_F_R'` and configure `file_path` and `target_byte` accordingly.

- ASCADv1 fixed key: https://github.com/ANSSI-FR/ASCAD
- ASCADv1 random key: https://github.com/ANSSI-FR/ASCAD

### Dataset `TRACE_USIM` (provided)

Raw traces collected from a real USIM card, provided for reproducibility:

https://www.dropbox.com/scl/fo/ewp7s8hrr2a6f6klemj7g/ALWPDFkDh69CzVnSTadq0mk?rlkey=qr5pvxmmfg4zl595fyuxm72sl&st=37ne8x1b&dl=0

The H5 file is expected to contain:
- `trace`: power/EM trace array, shape `(N, num_features)`
- `label_parallel`: intermediate values (before HW mapping), shape `(N, 256)`

---

## Requirements

```
Python >= 3.8
PyTorch >= 1.10
numpy
h5py
```

Install dependencies:

```bash
pip install torch numpy h5py
```

---

## Usage

1. Set `file_path` to your `.h5` dataset file.
2. Set `base_path` to the directory where results (`.npz` files) should be saved.
3. Adjust `target_byte`, `start_traces_idx`, `end_traces_idx`, and other hyperparameters as needed.
4. Run:

```bash
python end2end_DL_NSCA_framework/convWIN-mcr_raw_hw_gs_paras.py
```

The script sweeps over `stride` values `[200, 500, 1000, 2000]` and increasing trace counts, repeating each configuration `repeat_experi` times. Results are saved as `.npz` files containing per-channel loss history and training time.

---

## Output Format

Each saved `.npz` file contains:

- `losses`: tensor of shape `(256, epochs)` — per-channel MSE loss over training epochs
- `time`: total training time in seconds

File naming convention:
```
convWIN-MCR_raw_{target}_hw_numft{num_features}_numtr{num_traces}_s{stride}_k{kernel}_{sl1}_{sl2}_{lr}_{bs}_{epochs}epoch_{j}experi.npz
```

---

## Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@inproceedings{yu2025end,
  title={End-to-End Non-profiled Side-Channel Analysis on Long Raw Traces},
  author={Yu, Jintong and Wang, Yuxuan and Qu, Shipei and Zhao, Yubo and Shi, Yipeng and Cao, Pei and Lu, Xiangjun and Zhang, Chi and Gu, Dawu and Hong, Cheng},
  booktitle={European Symposium on Research in Computer Security},
  pages={526--544},
  year={2025},
  organization={Springer}
}
```

---

## Contact

For questions about the code or paper, please open an issue in this repository.
