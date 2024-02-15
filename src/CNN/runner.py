from dataset import MagnaTagATune
from torch import nn
import os
import torch
import argparse
from pathlib import Path
from multiprocessing import cpu_count
from torch.utils.tensorboard import SummaryWriter
from model import Model
from trainer import Trainer
from datetime import datetime


DATA_PATH = os.path.join("data", "MagnaTagATune")

log_dir = os.path.join(".", "logs")

parser = argparse.ArgumentParser(
    description="Train the coursework model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--dataset-root", default=DATA_PATH)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=0.005, type=float, help="Learning rate")
parser.add_argument("--momentum", default=0.99, type=float, help="Momentum")
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
    default=1,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=100,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=100,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=1,
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
parser.add_argument(
    "--save",
    default=False,
    type=bool,
    help="Whether the final model should be saved."
)


def get_summary_writer_log_dir(args) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        The path to the log directory.
    """

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    i = 0
    tb_log_dir = f'CNN_bs={args.batch_size}_lr={args.learning_rate}_mom={args.momentum}_{i}'
    while tb_log_dir in os.listdir(log_dir):
        i += 1
        tb_log_dir = f'CNN_bs={args.batch_size}_lr={args.learning_rate}_mom={args.momentum}_{i}'

    tb_log_dir_path = os.path.join(log_dir, tb_log_dir)
    return str(tb_log_dir_path)


def main(args):
    # Load the train, validation and test datasets and create
    train_labels_path = os.path.join(args.dataset_root, "annotations", "new_train_labels.pkl")
    val_labels_path = os.path.join(args.dataset_root, "annotations", "new_val_labels.pkl")
    test_labels_path = os.path.join(args.dataset_root, "annotations", "new_test_labels.pkl")
    samples_path = os.path.join(args.dataset_root, "samples")
    train_dataset = MagnaTagATune(train_labels_path, samples_path)
    val_dataset = MagnaTagATune(val_labels_path, samples_path)
    test_dataset = MagnaTagATune(test_labels_path, samples_path)

    torch.set_default_tensor_type(torch.HalfTensor)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
        drop_last = True
    )
    # Unshuffled train loader for AUC score computation on train dataset
    train_loader2 = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    # Move data to appropriate device
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        DEVICE = torch.device("cpu")
    print(f'Running model on device {DEVICE}')

    # Define the model, criterion and optimizer
    print(f'Defining model with learning rate {args.learning_rate}, momentum {args.momentum}.')
    model = Model(args.length, args.stride)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum)

    # Initialise logging
    log_path = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_path}")
    summary_writer = SummaryWriter(log_path, flush_secs=5)

    model_path = f"{log_path}_model.pth"
    if args.save:
        print(f"Saving final state of the model to {model_path}")
    else:
        print("Model will not be saved. Use the flag --save True to save the model.")

    # Define trainer and train the model
    trainer = Trainer(
        model, train_loader, train_loader2, val_loader, test_loader, train_labels_path, val_labels_path, test_labels_path, criterion, optimizer, summary_writer, DEVICE
    )
    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    if args.save:
        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main(parser.parse_args())