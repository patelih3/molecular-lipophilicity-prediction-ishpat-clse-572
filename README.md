# Molecular Lipophilicity Prediction

This package trains machine learning models to predict molecular lipophilicity using molecular fingerprints (Morgan fingerprints and MACCS keys).

## Installation

1. Clone this repository:
```bash
git clone https://github.com/patelih3/molecular-lipophilicity-prediction-ishpat-clse-572.git
cd molecular-lipophilicity-prediction-ishpat-clse-572
```

2. Create the conda environment:
```bash
conda env create -f environment.yml
conda activate lipophilicity-env
```

## Usage

Navigate to the package folder and run the training script:
```bash
cd lipophilicity_models
python train_models.py
```

This will train two models (Morgan fingerprints and MACCS keys) and display their RMSE scores.

## Dataset

The lipophilicity dataset contains molecular SMILES strings and experimental lipophilicity values.

## Requirements

- Python 3.9
- RDKit
- scikit-learn
- pandas
- numpy
```