from utils.dataset import MagnaTagATune
from torch import nn
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from model import Model
from trainer import Trainer


# DATA_PATH = os.path.join("data", "MagnaTagATune", "samples")
# TRAIN_LABELS_PATH = os.path.join("data", "MagnaTagATune", "annotations", "train_labels.pkl")
# VAL_LABELS_PATH = os.path.join("data", "MagnaTagATune", "annotations", "val_labels.pkl")

SCRATCH_DIR = os.path.join(os.sep, "mnt", "storage", "scratch", "wh20899")
DATA_PATH = os.path.join(SCRATCH_DIR, "MagnaTagATune", "samples")
TRAIN_LABELS_PATH = os.path.join(SCRATCH_DIR, "MagnaTagATune", "annotations", "train_labels.pkl")
VAL_LABELS_PATH = os.path.join(SCRATCH_DIR, "MagnaTagATune", "annotations", "val_labels.pkl")

batch_size = 10
worker_count = 4
learning_rate = 0.1
epochs = 5
val_frequency = 5
print_frequency = 10
log_frequency = 10
log_dir = os.path.join(".", "logs")


def get_summary_writer_log_dir() -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'CNN_bs={batch_size}_lr={learning_rate}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = os.path.join(log_dir, (tb_log_dir_prefix + str(i)))
        if os.path.exists(tb_log_dir):
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


def main():

    train_dataset = MagnaTagATune(TRAIN_LABELS_PATH, DATA_PATH)
    test_dataset = MagnaTagATune(VAL_LABELS_PATH, DATA_PATH)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=worker_count,
        pin_memory=True,
    )

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    model = Model()

    criterion = nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), learning_rate)

    log_dir = get_summary_writer_log_dir()
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE
    )

    trainer.train(
        epochs,
        val_frequency,
        print_frequency=print_frequency,
        log_frequency=log_frequency,
    )

if __name__ == "__main__":
    main()