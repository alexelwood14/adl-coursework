from utils.dataset import MagnaTagATune
from torch import nn
import os
import torch
import argparse
from pathlib import Path
from multiprocessing import cpu_count
from torch.utils.tensorboard import SummaryWriter
from model import Model
from trainer import Trainer


DATA_PATH = os.path.join("data", "MagnaTagATune", "samples")
TRAIN_LABELS_PATH = os.path.join("data", "MagnaTagATune", "annotations", "train_labels.pkl")
VAL_LABELS_PATH = os.path.join("data", "MagnaTagATune", "annotations", "val_labels.pkl")

# SCRATCH_DIR = os.path.join(os.sep, "mnt", "storage", "scratch", "wh20899")
# DATA_PATH = os.path.join(SCRATCH_DIR, "MagnaTagATune", "samples")
# TRAIN_LABELS_PATH = os.path.join(SCRATCH_DIR, "MagnaTagATune", "annotations", "train_labels.pkl")
# VAL_LABELS_PATH = os.path.join(SCRATCH_DIR, "MagnaTagATune", "annotations", "val_labels.pkl")

log_dir = os.path.join(".", "logs")

parser = argparse.ArgumentParser(
    description="Train the cousework model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-2, type=float, help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=10,
    type=int,
    help="Number of auto clip sets within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=20,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument(
    "--stride",
    default=256,
    type=int,
    help="Stride of convolutional filter",
)
parser.add_argument(
    "--length",
    default=256,
    type=int,
    help="Length of convolutional filter",
)


def get_summary_writer_log_dir(args) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'CNN_bs={args.batch_size}_lr={args.learning_rate}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = os.path.join(log_dir, (tb_log_dir_prefix + str(i)))
        if os.path.exists(tb_log_dir):
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


def main(args):
    train_dataset = MagnaTagATune(TRAIN_LABELS_PATH, DATA_PATH)
    test_dataset = MagnaTagATune(VAL_LABELS_PATH, DATA_PATH)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    model = Model(args.length, args.stride)

    criterion = nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

if __name__ == "__main__":
    main(parser.parse_args())