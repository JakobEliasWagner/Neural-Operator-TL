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

To evaluate the performance run
```shell
python trial/evaluate.py --run-dir=<dir>
```
using the correct run dir, containing models.

## Visualize
To visualize the operators (csv)
```shell
python trial/visualize.py --run-dir=<dir>
```
using the correct run dir, containing the trained models.
