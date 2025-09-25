"""Data loading utilities following EffectiveConPerturb project standards"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Project constants following your existing patterns
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATAPATH = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def load_metadata(
    datapath: str = DATAPATH,
    metadata_file: str = None,
    columns: List[str] = None,
    sc_root: str = None
) -> pd.DataFrame:
    """Read metadata file and return specified columns, or create IPP dictionary from SC matrices.
    
    Parameters
    ----------
    datapath : str
        Base data directory.
    metadata_file : str, optional
        Name of the metadata file (e.g., "MY_METADATA.csv"). If None, will extract IPPs from SC matrices.
    columns : List[str], optional
        Columns to extract from metadata. If None and metadata_file is provided, returns all columns.
        If metadata_file is None, this parameter is ignored.
    sc_root : str, optional
        SC folder name under data/SCs/ (should come from config file). Required when metadata_file is None.
        
    Returns
    -------
    pd.DataFrame
        Either the metadata with specified columns or a DataFrame with IPP column from SC matrices.
    """
    # If metadata file is provided, load it
    if metadata_file is not None:
        print(f"Loading metadata from: {metadata_file}")
        meta_path = os.path.join(datapath, "metadata", metadata_file)
        if os.path.exists(meta_path):
            meta = pd.read_csv(meta_path, dtype=str)
            # Apply standard completion filter if "Done" column exists
            if "Done" in meta.columns:
                meta = meta[meta["Done"] == "1"]
            
            # Return specified columns or all columns
            if columns is not None:
                available_cols = [col for col in columns if col in meta.columns]
                if len(available_cols) != len(columns):
                    missing_cols = [col for col in columns if col not in meta.columns]
                    print(f"Warning: Missing columns in metadata: {missing_cols}")
                return meta[available_cols] if available_cols else meta
            else:
                return meta
        else:
            print(f"Warning: Metadata file {meta_path} not found. Extracting IPPs from SC matrices.")
    
    # Fallback: Extract IPPs from SC matrices
    if sc_root is None:
        raise ValueError("sc_root must be provided when no metadata file is available (should come from config)")
    
    print(f"Creating IPP list from SC matrices in: {sc_root}")
    sc_dir = os.path.join(datapath, "SCs", sc_root)
    
    if not os.path.exists(sc_dir):
        raise FileNotFoundError(f"SC directory not found: {sc_dir}")
    
    ipp_set = set()
    
    # Check if sc_root has vendor subfolders (like workbench_siemens, workbench_GE)
    potential_vendors = [item for item in os.listdir(sc_dir) 
                        if os.path.isdir(os.path.join(sc_dir, item))]
    
    has_csv_files_in_root = any(f.endswith('.csv') for f in os.listdir(sc_dir) 
                               if os.path.isfile(os.path.join(sc_dir, f)))
    
    if has_csv_files_in_root:
        # CSV files are directly in sc_root
        for file in os.listdir(sc_dir):
            if file.endswith('.csv'):
                ipp = file[:-4]  # Remove .csv extension
                ipp_set.add(ipp)
    else:
        # Look for CSV files in vendor subfolders
        for vendor in potential_vendors:
            vendor_dir = os.path.join(sc_dir, vendor)
            if os.path.exists(vendor_dir):
                for file in os.listdir(vendor_dir):
                    if file.endswith('.csv'):
                        ipp = file[:-4]  # Remove .csv extension
                        ipp_set.add(ipp)
    
    if not ipp_set:
        raise FileNotFoundError(f"No CSV files found in {sc_dir} or its subdirectories")
    
    # Create DataFrame with IPP column
    ipp_list = sorted(list(ipp_set))
    print(f"Found {len(ipp_list)} IPPs from SC matrices")
    return pd.DataFrame({"IPP": ipp_list})

def split_groups(meta: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (VS_IPPs, MCS_IPPs) lists from metadata."""
    vs = meta[meta["CRS"] == "VS"]["IPP"].tolist()
    mcs = (
        meta[meta["CRS"] == "MCS-"]["IPP"].tolist() +
        meta[meta["CRS"] == "MCS+"]["IPP"].tolist()
    )
    return vs, mcs

def load_sc_matrix(
    ipp: str,
    datapath: str = DATAPATH,
    sc_root: str = None,
    normalize: bool = True
) -> np.ndarray:
    """Return the SC matrix (NumPy) following project normalization: C = C/max(C(:))
    
    Parameters
    ----------
    ipp : str
        Patient identifier.
    datapath : str
        Base data directory.
    sc_root : str
        SC folder name under data/SCs/ (should come from config file).
    normalize : bool
        Whether to apply max-normalization.
    """
    if sc_root is None:
        raise ValueError("sc_root must be provided (should come from config)")
    
    sc_dir = os.path.join(datapath, "SCs", sc_root)
    
    # First check if CSV is directly in sc_root
    direct_path = os.path.join(sc_dir, f"{ipp}.csv")
    if os.path.exists(direct_path):
        mat = pd.read_csv(direct_path, header=None).values
        if normalize:
            mat = 0.2 * mat / max(mat.max(), 1e-6) # DMF-specific normalization
        return mat
    
    # If not found directly, look in vendor subfolders
    if os.path.exists(sc_dir):
        for vendor in os.listdir(sc_dir):
            vendor_path = os.path.join(sc_dir, vendor)
            if os.path.isdir(vendor_path):
                csv_path = os.path.join(vendor_path, f"{ipp}.csv")
                if os.path.exists(csv_path):
                    mat = pd.read_csv(csv_path, header=None).values
                    if normalize:
                        mat = 0.2 * mat / max(mat.max(), 1e-6)
                    return mat

    raise FileNotFoundError(f"SC matrix for {ipp} not found in {sc_dir} or its subdirectories")

def load_all_sc_matrices(
    ipp_list: List[str],
    datapath: str = DATAPATH,
    sc_root: str = None,
) -> Dict[str, np.ndarray]:
    """Load all SC matrices for given patient list.

    Parameters
    ----------
    ipp_list : List[str]
        Patient identifiers to load.
    datapath : str
        Base data directory.
    sc_root : str
        SC folder name under data/SCs/ (should come from config file).
    """
    if sc_root is None:
        raise ValueError("sc_root must be provided (should come from config)")
        
    matrices: Dict[str, np.ndarray] = {}
    for ipp in ipp_list:
        try:
            matrices[ipp] = load_sc_matrix(ipp, datapath=datapath, sc_root=sc_root)
        except FileNotFoundError as e:
            print(str(e))  # Warn but continue
    return matrices

def load_bootstrapped_matrices(datapath: str = DATAPATH, nregions: int = 100) -> Dict[str, np.ndarray]:
    """Load bootstrap SC matrices following project preprocessing patterns"""
    bootstrapped_patients = {}
    bootstrap_dir = os.path.join(datapath,"SCs", "Bootstrapped_SCs")

    for patient_folder in os.listdir(bootstrap_dir):
        folder_path = os.path.join(bootstrap_dir, patient_folder)
        if not os.path.isdir(folder_path):
            continue
            
        print(f"Patient folder: {patient_folder}")
        bootstrapped_SCs = np.zeros((50, nregions, nregions))
        
        for it in range(1, 51):
            fname = os.path.join(folder_path, f"SC_sift_{it:02d}.csv")
            if os.path.exists(fname):
                # Project-specific preprocessing
                raw_data = pd.read_csv(fname, header=None).values[1:, 1:]  # Skip header row/col
                cleaned = raw_data - np.eye(nregions) * raw_data  # Remove diagonal
                bootstrapped_SCs[it-1] = cleaned / max(cleaned.max(), 1e-6)  # Standard normalization
        
        bootstrapped_patients[patient_folder] = bootstrapped_SCs
    
    return bootstrapped_patients

def get_scanner_mapping(
    ipp_list: List[str],
    datapath: str = DATAPATH,
    sc_root: str = None,
) -> Dict[str, str]:
    """Map patient IDs to scanner vendors (e.g., Siemens/GE) by file presence.
    Only works when SC files are organized in vendor subdirectories.
    """
    if sc_root is None:
        raise ValueError("sc_root must be provided (should come from config)")
        
    sc_dir = os.path.join(datapath, "SCs", sc_root)
    scanner_map: Dict[str, str] = {}
    
    if not os.path.exists(sc_dir):
        return scanner_map
    
    # Get list of potential vendor folders
    potential_vendors = [item for item in os.listdir(sc_dir) 
                        if os.path.isdir(os.path.join(sc_dir, item))]
    
    for ipp in ipp_list:
        for vendor in potential_vendors:
            vendor_path = os.path.join(sc_dir, vendor, f"{ipp}.csv")
            if os.path.exists(vendor_path):
                scanner_map[ipp] = vendor
                break
    
    return scanner_map