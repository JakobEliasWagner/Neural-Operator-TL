# Neural-Operator-TL

## Installing Dependencies

To install the dependencies install them using Poetry by running:
```shell
poetry install
```

## Downloading the Datasets

The dataset is available on [huggingface](https://huggingface.co/datasets/JakobEWagner/transmission_loss).

To download the dataset using git make sure `git-lfs` is installed
```shell
git lfs install
```
Then run
```shell
git clone https://huggingface.co/datasets/JakobEWagner/transmission_loss data
```

## Train Operators

To train all models we use Hydra.
Run
```shell
python trial/run.py --multirun
```
to run the configuration we used.

## Evaluate Performance


## Visualize
