#!/bin/bash

# Rename annotations file
mv $1/annotations/val_labels.pkl $1/annotations/test_labels.pkl 
echo "Annotations renamed"

# Split samples into train and val
mv $1/samples/val $1/samples/test
mkdir $1/samples/val
mv $1/samples/train/c $1/samples/val/
echo "Train split into train and validation"