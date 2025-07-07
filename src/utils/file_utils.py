"""
File handling utility functions for the medical image analysis project.
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple


def collect_files(base_dir: str, extensions: List[str] = None) -> List[str]:
    """
    Collect files from a directory with specified extensions.
    
    Args:
        base_dir (str): Base directory to search for files
        extensions (List[str]): List of file extensions to include (e.g., ['.dcm', '.nii'])
        
    Returns:
        List[str]: List of file paths found
    """
    if extensions is None:
        extensions = ['.dcm', '.nii', '.nii.gz']
    
    files = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Warning: Directory {base_dir} does not exist")
        return files
    
    for ext in extensions:
        files.extend(list(base_path.rglob(f"*{ext}")))
    
    return [str(f) for f in files]


def generate_dataframe(included_files: List[str]) -> pd.DataFrame:
    """
    Generate a DataFrame from a list of file paths with metadata.
    
    Args:
        included_files (List[str]): List of file paths
        
    Returns:
        pd.DataFrame: DataFrame with file information
    """
    data = []
    
    for file_path in included_files:
        path = Path(file_path)
        data.append({
            'file_path': str(path),
            'filename': path.name,
            'directory': str(path.parent),
            'size_bytes': path.stat().st_size if path.exists() else 0,
            'extension': path.suffix
        })
    
    return pd.DataFrame(data)


def save_qa_report(total_files: int, included_count: int, excluded_count: int, 
                   output_path: str = "data_ingestion_QA_report.csv") -> None:
    """
    Save a quality assurance report for data ingestion.
    
    Args:
        total_files (int): Total number of files found
        included_count (int): Number of files included
        excluded_count (int): Number of files excluded
        output_path (str): Path to save the QA report
    """
    qa_data = {
        'metric': ['total_files', 'included_files', 'excluded_files', 'inclusion_rate'],
        'value': [total_files, included_count, excluded_count, 
                 f"{(included_count/total_files)*100:.2f}%" if total_files > 0 else "0%"]
    }
    
    qa_df = pd.DataFrame(qa_data)
    qa_df.to_csv(output_path, index=False)
    print(f"QA report saved to {output_path}") 