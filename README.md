# COMSM0045 Applied Deep Learning Coursework

## Train, Val and Test Split

Before training the models, the data must reformatted so that there are 12 training, 1 validation and 3 testing parts.

From the project root directory, navigate to the scripts directory.

```bash
cd src/scripts
```

Run the splitdata.sh script and pass in the path to the MagnaTagATune dataset.

```bash
./splitdata.sh <path-to-magnatagatune>
```

Change directory to utils

```bash
cd ../utils
```

Run python script to update annotation files to reflect changes to train, val, test split.

```bash
python create_test_val.py --annotations-path=<path-to-magnatagatune-annotations>
```

Within the MagnaTagATune dataset samples directory, there should be train, val and test directories. 

## Running CNN

Our model which replicates the CNN architecture from the Dieleman and Schrauwen can be found in [src/CNN/](src/CNN/).

From the project root directory, navigate to model directory

```bash
cd src/CNN
```
Edit Line 18 of `train.sh` with correct data path and desired hyperparameters. 

To run locally use:

```bash
./train.sh
```
To run on BC4 with Slurm:

```bash
sbatch train.sh
```

## Running CRNN

Our improved model which replicates the CRNN architecture from Choi et al. can be found in [src/CRNN/](src/CRNN/).

Before running this model. The spectrogram data must be generated from the original raw audio signals. To do this, run:

```bash
python src/utils/spectrogram.py <path-to-magnatagatune>
```

From the project root directory, navigate to model directory

```bash
cd src/CRNN
```
Edit Line 18 of `train.sh` with correct data path and desired hyperparameters. 

To run locally use:

```bash
./train.sh
```
To run on BC4 with Slurm:

```bash
sbatch train.sh
```