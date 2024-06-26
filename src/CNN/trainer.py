from torch import nn
from typing import Union
from torch.optim.optimizer import Optimizer
import time, argparse, torch
from torch.utils.data import DataLoader
from evaluation import evaluate
import numpy as np

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        train_loader2: DataLoader, # Unshuffled for validation
        val_loader: DataLoader,
        test_loader: DataLoader,
        train_path: str,
        val_path: str,
        test_path: str,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.train_loader2 = train_loader2
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for _, batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                logits = self.model.forward(batch)
                loss = self.criterion(logits, labels)

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels.argmax(-1), preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                # if ((self.step + 1) % log_frequency) == 0:
                #     self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            # self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                # Validate and draw AUC curves
                self.validate(epoch, self.train_path, self.train_loader2, "train")
                self.validate(epoch, self.val_path, self.val_loader, "val")

                # Draw accuracy and loss curves
                self.draw_curves(epoch, self.train_loader, "train")
                self.draw_curves(epoch, self.val_loader, "val")

                # Switch back to train mode
                self.model.train()

        # Evaluate against the test set
        self.validate(epoch, self.test_path, self.test_loader, "test")
        self.draw_curves(epoch, self.test_loader, "test")

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def draw_curves(self, epoch, data_loader, curve_type):
        self.model.eval()

        loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for _, batch, labels in data_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)

                # Compute loss
                loss += self.criterion(logits, labels)

                # Compute number of correct guesses
                labels = labels.argmax(-1)
                preds = logits.argmax(-1)
                correct += float((labels == preds).sum())
                total += len(labels)

        # Compute accuracy
        accuracy = correct / total
        loss = loss / total

        # Log accuracy and loss curves
        self.summary_writer.add_scalars(
                "accuracy",
                {curve_type: accuracy},
                epoch
        )
        self.summary_writer.add_scalars(
                "loss",
                {curve_type: loss},
                epoch
        )

             

    def validate(self, epoch, data_path, data_loader, curve_type):
        all_preds = []
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for _, batch, labels in data_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)

                # Collecting all preds
                preds = logits.cpu().numpy() # .argmax(dim=-1) removed
                all_preds.extend(list(preds))

        # AUC Evaluation
        all_preds = torch.tensor(np.array(all_preds)).to(self.device)
        auc = evaluate(all_preds, data_path)

        # Log for curves
        self.summary_writer.add_scalars(
                "AUC",
                {curve_type: auc},
                epoch
        )


def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
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
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)