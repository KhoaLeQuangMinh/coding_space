import os
import sys
import csv
import random
import datetime
import numpy as np
import torch


# ══════════════════════════════════════════════════════════════════════════════
# Tee  —  mirrors stdout to a file so every print() appears in the run log
# ══════════════════════════════════════════════════════════════════════════════

class Tee:
    """
    Redirect stdout to both the terminal and an open file handle.

    Usage
    -----
        tee = Tee(log_file_handle)
        sys.stdout = tee
        ...
        sys.stdout = tee.restore()   # put the original stdout back
    """

    def __init__(self, file_handle):
        self._file   = file_handle
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def restore(self):
        return self._stdout


# ══════════════════════════════════════════════════════════════════════════════
# Run-config snapshot  —  writes every CLI arg to the top of the run log
# ══════════════════════════════════════════════════════════════════════════════

def save_run_config(args, log_path: str) -> "Tee":
    """
    Create (or append to) a plain-text run log at `log_path`, write a
    header block with all CLI arguments and a timestamp, then attach a
    Tee so every subsequent print() also lands in that file.

    Returns the Tee instance; the caller should restore stdout when done:

        tee = save_run_config(args, log_path)
        ...
        sys.stdout = tee.restore()
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    log_file = open(log_path, "a")          # append so re-runs accumulate

    header = [
        "=" * 62,
        f"  RUN  —  {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
        "=" * 62,
        "  ARGUMENTS",
        "-" * 62,
    ]
    for k, v in vars(args).items():
        header.append(f"  {k:<30} {v}")
    header += ["-" * 62, "  TRAINING LOG", "-" * 62, ""]

    block = "\n".join(header) + "\n"
    log_file.write(block)
    log_file.flush()

    tee = Tee(log_file)
    sys.stdout = tee
    return tee


# ══════════════════════════════════════════════════════════════════════════════
# CSV experiment logger  (unchanged API)
# ══════════════════════════════════════════════════════════════════════════════

def create_experiment_logger(experiment_name: str):
    os.makedirs("outputs/logs", exist_ok=True)
    log_path    = f"outputs/logs/{experiment_name}.csv"
    file_exists = os.path.exists(log_path)
    file        = open(log_path, mode="a", newline="")
    writer      = csv.writer(file)

    if not file_exists:
        writer.writerow([
            "epoch", "train_loss", "val_loss",
            "accuracy", "macro_f1", "macro_recall", "lr",
        ])

    return file, writer


# ══════════════════════════════════════════════════════════════════════════════
# Pretty-print a flat args namespace
# ══════════════════════════════════════════════════════════════════════════════

def print_experiment_config(args):
    mode = getattr(args, 'training_mode', 'standard')
    model = getattr(args, 'model_type', 'fusion')
    loss = getattr(args, 'loss', 'crossentropy')
    print("\n" + "=" * 60)
    print(f"  Experiment : {args.experiment_name}")
    print("-" * 60)
    print(f"  Mode       : {mode}")
    print(f"  Model      : {model}")
    if model == 'fusion':
        print(f"  Fusion     : {args.fusion_type}")
    print(f"  Loss       : {loss}")
    print(f"  Classes    : {args.num_classes}")
    print("-" * 60)
    print(f"  Device     : {args.device}")
    print(f"  Epochs     : {args.epochs}")
    print("-" * 60)
    print(f"  LR         : {args.lr}")
    print(f"  Weight dec.: {args.weight_decay}")
    print(f"  Momentum   : {args.momentum}")
    print("-" * 60)
    print(f"  Sched T0   : {args.T_0}")
    print(f"  Sched Tmul : {args.T_mult}")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Reproducibility helpers  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def set_global_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic    = True
        torch.backends.cudnn.benchmark        = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Fallback for very old torch versions (<1.11) where warn_only was not present
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)