"""
Ishan Patel
VCU - AI/ML - Fall 2025
HW5 - Python Package

"""

"""
Functions for generating molecular fingerprints from SMILES strings.
"""

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
import numpy as np


def get_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    """
    Generate Morgan fingerprints from SMILES strings.
    
    Parameters:
    -----------
    smiles_list : list
        List of SMILES strings
    radius : int
        Radius for Morgan fingerprint (default=2)
    n_bits : int
        Number of bits in fingerprint (default=2048)
    
    Returns:
    --------
    fps : np.array
        Array of fingerprints
    valid_idx : list
        Indices of valid molecules
    """
    fps = []
    valid_idx = []
    
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp))
            valid_idx.append(i)
    
    return np.array(fps), valid_idx


def get_maccs_keys(smiles_list):
    """
    Generate MACCS keys from SMILES strings.
    
    Parameters:
    -----------
    smiles_list : list
        List of SMILES strings
    
    Returns:
    --------
    fps : np.array
        Array of MACCS key fingerprints
    valid_idx : list
        Indices of valid molecules
    """
    fps = []
    valid_idx = []
    
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            fps.append(np.array(fp))
            valid_idx.append(i)
    
    return np.array(fps), valid_idx