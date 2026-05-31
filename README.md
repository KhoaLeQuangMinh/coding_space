# Multimodal MRI+PET Alzheimer Classification & HOPE Baseline

This repository contains the code for classifying Alzheimer's Disease using paired MRI and PET scans.
It includes the standard Multimodal Baselines (MRI-only, PET-only, and Fusion) as well as an exact replication of the **HOPE (Hybrid-granularity Ordinal Prototype lEarning)** baseline for mild cognitive impairment prediction.

---

## 1. Installation

Ensure you have Python 3.9+ and PyTorch installed. Install the dependencies:
```bash
pip install -r requirements.txt
```
*(If you are running on Kaggle, most dependencies like `torch`, `scipy`, and `nibabel` are already available.)*

---

## 2. Usage Guide (Training & Testing)

We have refactored the pipeline to make argument mismatch impossible. You just train the model, and the configuration is saved securely in `outputs/runs/<experiment_name>/args.json`.

### Option A: Standard Baseline (MRI / PET / Fusion)
Use this mode to train the old standard baselines. It predicts 4 classes (CN, sMCI, pMCI, AD).

**Train a Fusion Model (Concat + CrossEntropy)**
```bash
python train.py \
    --experiment_name fusion_baseline \
    --training_mode standard \
    --model_type fusion \
    --fusion_type concat \
    --loss crossentropy \
    --data_root /path/to/data \
    --epochs 40 \
    --batch_size 4
```

**Train an MRI-Only Model (Focal Loss)**
```bash
python train.py \
    --experiment_name mri_only_focal \
    --training_mode standard \
    --model_type mri_only \
    --loss focal \
    --data_root /path/to/data
```

### Option B: Exact HOPE Baseline
Use this mode to perfectly replicate the HOPE paper (1-channel 3D ResNet for MRI). 
It automatically sets the loss to the custom `HopeLossCriterion`, handles the 3-class proportional batching (`HopeBatchSampler`), and prepares the EMA Prototypes.

```bash
python train.py \
    --experiment_name hope_replication \
    --training_mode hope \
    --data_root /path/to/data \
    --epochs 60 \
    --batch_size 8 \
    --lr 0.001
```
*Note: Any arguments like `--loss mse` or `--model_type fusion` will be safely overridden when `--training_mode hope` is selected.*

---

## 3. Unified Post-Training Analysis

You no longer have to pass the exact same arguments for analysis! The unified `analyse.py` script automatically loads your settings and runs the correct logic.

```bash
python analyse.py --experiment_name <your_experiment_name>
```

**What it does:**
- If it was a **Standard** run: Plots the Loss/Metrics curves and generates the standard 4-class Confusion Matrix.
- If it was a **HOPE** run: Plots the curves, evaluates the 3-class classifier head (CN vs MCI vs AD), AND evaluates the custom 4-class Prototype Similarity to split sMCI and pMCI, generating a final 4-class Confusion Matrix to compare against the standard baseline.

Outputs are saved directly into your experiment's folder (`outputs/runs/<experiment_name>/`).

---

## 4. Local Testing on Mac (Mock Data)

If you do not have the real `.npz` dataset on your local machine, you can append the `--mock_data` flag. This will generate random `128x128x128` tensors, allowing you to quickly test your pipeline changes before pushing to Kaggle.

```bash
python train.py --experiment_name test_local --training_mode hope --mock_data --epochs 1 --batch_size 4
python analyse.py --experiment_name test_local --mock_data
```

---

## 5. K-Fold Cross Validation

Append `--kfold 5` to `train.py` to run 5-fold cross-validation. It holds out a final test set, trains 5 models on the remaining data, prints the `Mean ± Std` of the Validation F1, and saves checkpoints individually.