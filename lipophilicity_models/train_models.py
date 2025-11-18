"""
Ishan Patel
VCU - AI/ML - Fall 2025
HW5 - Python Package

"""

"""
Train lipophilicity prediction models using molecular fingerprints.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import fingerprint functions from our module
from fingerprints import get_morgan_fp, get_maccs


def main():
    """Main function to train and evaluate models."""
    
    # Get conda environment name
    conda_env = os.getenv("CONDA_DEFAULT_ENV")
    if conda_env is None:
        conda_env = "Environment name not found"
    
    print("="*60)
    print("Lipophilicity Prediction Model Training")
    print("="*60)
    print(f"\nConda Environment: {conda_env}\n")
    
    # Load data
    print("Loading dataset...")
    data = pd.read_csv('Lipophilicity.csv')
    print(f"Dataset loaded: {len(data)} molecules")
    
    target_col = 'exp'
    
    # Split data
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Train set: {len(train_df)} molecules")
    print(f"Test set: {len(test_df)} molecules\n")
    
    # Generate Morgan fingerprints using imported function
    print("Generating Morgan fingerprints...")
    train_morgan, train_morgan_idx = get_morgan_fp(train_df['smiles'].tolist())
    test_morgan, test_morgan_idx = get_morgan_fp(test_df['smiles'].tolist())
    print(f"Morgan FP shape: {train_morgan.shape}")
    
    # Generate MACCS keys using imported function
    print("Generating MACCS keys...")
    train_maccs, train_maccs_idx = get_maccs(train_df['smiles'].tolist())
    test_maccs, test_maccs_idx = get_maccs(test_df['smiles'].tolist())
    print(f"MACCS shape: {train_maccs.shape}\n")
    
    # Get targets
    train_y_morgan = train_df.iloc[train_morgan_idx][target_col].values
    test_y_morgan = test_df.iloc[test_morgan_idx][target_col].values
    train_y_maccs = train_df.iloc[train_maccs_idx][target_col].values
    test_y_maccs = test_df.iloc[test_maccs_idx][target_col].values
    
    # Scale targets
    scaler_morgan = StandardScaler()
    train_y_morgan_scaled = scaler_morgan.fit_transform(train_y_morgan.reshape(-1, 1)).flatten()
    
    scaler_maccs = StandardScaler()
    train_y_maccs_scaled = scaler_maccs.fit_transform(train_y_maccs.reshape(-1, 1)).flatten()
    
    # Train Morgan model
    print("Training Morgan fingerprint model...")
    model_morgan = MLPRegressor(
        hidden_layer_sizes=(100, 50), 
        max_iter=500, 
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model_morgan.fit(train_morgan, train_y_morgan_scaled)
    print(f"Morgan model done - {model_morgan.n_iter_} iterations")
    
    # Train MACCS model
    print("Training MACCS keys model...")
    model_maccs = MLPRegressor(
        hidden_layer_sizes=(100, 50), 
        max_iter=500, 
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model_maccs.fit(train_maccs, train_y_maccs_scaled)
    print(f"MACCS model done - {model_maccs.n_iter_} iterations\n")
    
    # Make predictions
    pred_morgan_scaled = model_morgan.predict(test_morgan)
    pred_morgan = scaler_morgan.inverse_transform(pred_morgan_scaled.reshape(-1, 1)).flatten()
    
    pred_maccs_scaled = model_maccs.predict(test_maccs)
    pred_maccs = scaler_maccs.inverse_transform(pred_maccs_scaled.reshape(-1, 1)).flatten()
    
# Calculate RMSE (using sqrt for older sklearn versions)
rmse_morgan = np.sqrt(mean_squared_error(test_y_morgan, pred_morgan))
r2_morgan = r2_score(test_y_morgan, pred_morgan)

rmse_maccs = np.sqrt(mean_squared_error(test_y_maccs, pred_maccs))
r2_maccs = r2_score(test_y_maccs, pred_maccs)
    
    # Print results
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Conda Environment: {conda_env}")
    print(f"Morgan - RMSE: {rmse_morgan:.4f}, R2: {r2_morgan:.4f}")
    print(f"MACCS  - RMSE: {rmse_maccs:.4f}, R2: {r2_maccs:.4f}")
    print("="*60)
    
    # Comparison
    if rmse_morgan < rmse_maccs:
        improvement = ((rmse_maccs - rmse_morgan) / rmse_maccs * 100)
        print(f"\nMorgan fingerprints performed better by {improvement:.1f}%")
    else:
        improvement = ((rmse_morgan - rmse_maccs) / rmse_morgan * 100)
        print(f"\nMACCS keys performed better by {improvement:.1f}%")


if __name__ == "__main__":
    main()