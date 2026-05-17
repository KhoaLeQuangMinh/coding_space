# Coding_Space
Here are all the commands for every script in the codebase.
---

## Training

```bash
# Minimal — only required args
python train.py \
  --experiment_name  mri_pet_concat_ce \
  --data_root        /data/paired_npz

# Full — every option explicit
python train.py \
  --experiment_name  mri_pet_concat_ce \
  --data_root        /data/paired_npz \
  --pretrained_path  /path/to/ViT_pretrained.pth.tar \
  --pretrained \
  --fusion_type      concat \
  --loss             crossentropy \
  --label_smoothing  0.1 \
  --num_classes      4 \
  --feature_dim      768 \
  --class_names      CN sMCI pMCI AD \
  --train_ratio      0.7 \
  --val_ratio        0.1 \
  --batch_size       4 \
  --num_workers      4 \
  --device           cuda:0 \
  --epochs           40 \
  --seed             12345 \
  --lr               0.001 \
  --weight_decay     0.001 \
  --momentum         0.9 \
  --T_0              10 \
  --T_mult           3 \
  --eta_min          0.00001
```

**Other loss variants** — swap `--loss` and add the relevant flag:
```bash
# MSE regression head
python train.py \
  --experiment_name  mri_pet_concat_mse \
  --data_root        /data/paired_npz \
  --loss             mse

# Focal loss (good for class-imbalanced data)
python train.py \
  --experiment_name  mri_pet_concat_focal \
  --data_root        /data/paired_npz \
  --loss             focal \
  --focal_gamma      2.0
```

**Fusion type variants** — everything else stays the same, only `--fusion_type` changes:
```bash
--fusion_type  concat          # default: concatenate then linear
--fusion_type  sum             # element-wise sum
--fusion_type  film            # Feature-wise Linear Modulation
--fusion_type  gated           # Gated fusion
--fusion_type  CrossAttention  # bidirectional cross-attention
```

---

## Analysis (after training)

```bash
# Minimal
python analyse.py \
  --experiment_name  mri_pet_concat_ce \
  --data_root        /data/paired_npz

# Full
python analyse.py \
  --experiment_name  mri_pet_concat_ce \
  --data_root        /data/paired_npz \
  --loss             crossentropy \
  --fusion_type      concat \
  --num_classes      4 \
  --feature_dim      768 \
  --class_names      CN sMCI pMCI AD \
  --train_ratio      0.7 \
  --val_ratio        0.1 \
  --batch_size       4 \
  --num_workers      4 \
  --split            test \
  --top_n            20 \
  --scan_samples     3 \
  --device           cuda:0 \
  --seed             12345
  --kfold            5 \
  --fold             3      # whichever fold was printed as "Best fold"

# Run on validation split instead of test
python analyse.py \
  --experiment_name  mri_pet_concat_ce \
  --data_root        /data/paired_npz \
  --split            val

# MSE experiment — loss must match what was used at training time
python analyse.py \
  --experiment_name  mri_pet_concat_mse \
  --data_root        /data/paired_npz \
  --loss             mse
```

---

## Visualize training curves

```bash
# Only needs the experiment name — reads outputs/logs/<name>.csv
python visualize.py --experiment_name  mri_pet_concat_ce
```

---

## Run everything end-to-end (bash script)

Save this as `run_all.sh` to reproduce a full experiment in one shot:

```bash
#!/bin/bash
set -e  # stop on any error

NAME="mri_pet_concat_ce"
DATA="/data/paired_npz"
PRETRAINED="/path/to/ViT_pretrained.pth.tar"

echo "========== TRAIN =========="
python train.py \
  --experiment_name  $NAME \
  --data_root        $DATA \
  --pretrained_path  $PRETRAINED \
  --pretrained \
  --fusion_type      concat \
  --loss             crossentropy \
  --epochs           40 \
  --lr               0.001 \
  --device           cuda:0

echo "========== ANALYSE (test split) =========="
python analyse.py \
  --experiment_name  $NAME \
  --data_root        $DATA \
  --loss             crossentropy \
  --fusion_type      concat \
  --split            test \
  --device           cuda:0

echo "========== VISUALIZE CURVES =========="
python visualize.py \
  --experiment_name  $NAME

echo "========== DONE =========="
echo "Run log   : outputs/runs/$NAME/run_config.txt"
echo "CSV log   : outputs/logs/$NAME.csv"
echo "Plots     : outputs/logs/${NAME}_loss_curve.png"
echo "Analysis  : outputs/analysis/$NAME/"
```

```bash
chmod +x run_all.sh
./run_all.sh
```

---

## Output file map

| Script | Output |
|---|---|
| `train.py` | `[experiment_name].pth` — best checkpoint |
| | `outputs/runs/<name>/run_config.txt` — all args + full training log |
| | `outputs/logs/<name>.csv` — per-epoch metrics |
| `visualize.py` | `outputs/logs/<name>_loss_curve.png` |
| | `outputs/logs/<name>_metrics_curve.png` |
| `analyse.py` | `outputs/analysis/<name>/confusion_matrix.png` |
| | `outputs/analysis/<name>/per_class_metrics.png` |
| | `outputs/analysis/<name>/confidence_distribution.png` (CE/Focal) |
| | `outputs/analysis/<name>/calibration.png` (CE/Focal) |
| | `outputs/analysis/<name>/mse_residual_distribution.png` (MSE) |
| | `outputs/analysis/<name>/error_analysis.png` |
| | `outputs/analysis/<name>/hardest_samples.png` |
| | `outputs/analysis/<name>/confused_*.png` |
| | `outputs/analysis/<name>/error_subjects.json` |
| | `outputs/analysis/<name>/summary_report.txt` |