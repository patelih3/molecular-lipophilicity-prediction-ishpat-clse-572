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


def get_morgan_fp(smiles_list):
    """Generate Morgan fingerprints from SMILES strings."""
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(np.array(fp))
            valid_idx.append(i)
    return np.array(fps), valid_idx


def get_maccs(smiles_list):
    """Generate MACCS keys from SMILES strings."""
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            fps.append(np.array(fp))
            valid_idx.append(i)
    return np.array(fps), valid_idx