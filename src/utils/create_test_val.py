import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(
    description="Train the coursework model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--annotations_path", default=os.path.join("data", "MagnaTagATune", "annotations"))
args = parser.parse_args()

TRAIN_LABELS_PATH = os.path.join(args.annotations_path, "train_labels.pkl")
VAL_LABELS_PATH = os.path.join(args.annotations_path, "val_labels.pkl")

def replace_train(string):
    string = string.lstrip("train")
    string = "val" + string
    return string

old_train_labels = pd.read_pickle(TRAIN_LABELS_PATH)
is_in_set_c = old_train_labels['file_path'].str.startswith(os.path.join('train', 'c'))
train_labels = old_train_labels[is_in_set_c == False]
val_labels = old_train_labels[is_in_set_c]
val_labels["file_path"] = val_labels["file_path"].map(replace_train)

pd.to_pickle(val_labels, os.path.join(args.annotations_path, 'new_train_labels.pkl'))
pd.to_pickle(train_labels, os.path.join(args.annotations_path, 'new_val_labels.pkl'))

