import os
from utils.logger import get_logger
import pandas as pd
import warnings
import pydicom
import numpy as np

# Initialize logger for this module
logger = get_logger("data_ingestion")


def collect_files(base_dir):
    """
    Recursively collects DICOM files only from the expected folders:
    - PPMI_Images_PD: Label "PD"
    - PPMI_Images_SWEDD: Label "SWEDD"
    - PPMI_Images_Cont: Label "Control"

    Excludes any file containing "br_raw" in its path and logs all skipped folders.

    :param base_dir: Base directory containing the Images folder.
    :return: (included_files, excluded_files)
             included_files: list of tuples (full_path, label)
             excluded_files: list of file paths that were excluded.
    """
    logger.info(f"Starting file collection from base directory: {base_dir}")
    included_files = []
    excluded_files = []

    # Define the expected folders and corresponding labels
    expected_folders = {
        "PPMI_Images_PD": "PD",
        "PPMI_Images_SWEDD": "SWEDD",
        "PPMI_Images_Cont": "Control"
    }
    
    logger.info(f"Looking for these folders: {', '.join(expected_folders.keys())}")

    # Check if base directory exists
    if not os.path.exists(base_dir):
        logger.error(f"Base directory does not exist: {base_dir}")
        return included_files, excluded_files
        
    # Iterate over immediate subdirectories in base_dir
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path) and folder in expected_folders:
            logger.info(f"Processing folder: {folder_path}")
            # Log the group we're processing
            group_label = expected_folders[folder]
            logger.info(f"Processing {folder} (Group: {group_label})")
            
            # Track files per group
            group_file_count = 0
            
            # Recursively traverse the expected folder
            for root, dirs, files in os.walk(folder_path):
                dicom_files = [f for f in files if f.endswith(".dcm")]
                if dicom_files:
                    logger.debug(f"Found {len(dicom_files)} DICOM files in {root}")
                    
                for file in files:
                    if file.endswith(".dcm"):
                        full_path = os.path.join(root, file)
                        # Exclude any file with "br_raw" in its full path
                        if "br_raw" in full_path:
                            excluded_files.append(full_path)
                            logger.info(f"Excluding raw file: {full_path}")
                        else:
                            included_files.append((full_path, expected_folders[folder]))
                            group_file_count += 1
            
            logger.info(f"Collected {group_file_count} valid DICOM files for group {group_label}")
        else:
            logger.info(f"Skipping folder: {folder_path}")

    # Log summary statistics
    total_included = len(included_files)
    total_excluded = len(excluded_files)
    logger.info(f"File collection complete. Included: {total_included}, Excluded: {total_excluded}")
    
    # Log distribution by group
    group_counts = {}
    for _, label in included_files:
        group_counts[label] = group_counts.get(label, 0) + 1
    
    for group, count in group_counts.items():
        logger.info(f"Group {group}: {count} files ({count/total_included:.1%} of total)")
        
    return included_files, excluded_files


def generate_dataframe(included_files):
    """
    Creates a DataFrame from the list of validated file paths.

    :param included_files: List of tuples (file_path, label)
    :return: DataFrame with columns 'file_path' and 'label'
    """
    logger.info(f"Generating DataFrame from {len(included_files)} validated files")
    
    if not included_files:
        logger.warning("No files to include in DataFrame")
        return pd.DataFrame(columns=["file_path", "label"])
        
    df = pd.DataFrame(included_files, columns=["file_path", "label"])
    
    # Log distribution information
    group_distribution = df["label"].value_counts()
    logger.info(f"Group distribution in DataFrame:\n{group_distribution}")
    
    return df


def save_qa_report(total_files, included_count, excluded_count, output_path="data_ingestion_QA_report.csv"):
    """
    Generates and saves a QA report of the file collection process.

    :param total_files: Total number of DICOM files encountered.
    :param included_count: Count of files included after filtering.
    :param excluded_count: Count of files excluded.
    :param output_path: File path for the QA report CSV.
    """
    logger.info(f"Generating QA report for {total_files} total files")
    
    exclusion_ratio = excluded_count / total_files if total_files > 0 else 0
    inclusion_ratio = included_count / total_files if total_files > 0 else 0
    
    qa_report = {
        "total_files": total_files,
        "included_files": included_count,
        "excluded_files": excluded_count,
        "inclusion_ratio": inclusion_ratio,
        "exclusion_ratio": exclusion_ratio,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Log QA metrics
    logger.info(f"QA Metrics - Total: {total_files}, Included: {included_count} ({inclusion_ratio:.1%}), Excluded: {excluded_count} ({exclusion_ratio:.1%})")
    
    qa_df = pd.DataFrame([qa_report])
    qa_df.to_csv(output_path, index=False)
    logger.info(f"QA report saved to {output_path}")

    if exclusion_ratio > 0.5:
        logger.warning(f"High proportion of raw files excluded: {exclusion_ratio:.2%}")
        warnings.warn(f"High proportion of raw files excluded: {exclusion_ratio:.2%}")
        
    return qa_report


def load_dicom(file_path):
    """
    Loads and processes a DICOM file:
    - Reads the file using pydicom.
    - Converts the pixel array to float32.
    - Applies RescaleSlope and RescaleIntercept if available.

    :param file_path: Path to the DICOM file.
    :return: Tuple (processed_pixel_array, dicom_metadata)
    """
    logger.debug(f"Loading DICOM file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"DICOM file not found: {file_path}")
        raise FileNotFoundError(f"DICOM file not found: {file_path}")
    
    try:
        # Attempt to read the DICOM file
        logger.debug(f"Reading DICOM data from {file_path}")
        ds = pydicom.dcmread(file_path)
        
        # Log basic DICOM metadata
        if hasattr(ds, 'PatientID'):
            logger.debug(f"PatientID: {ds.PatientID}")
        if hasattr(ds, 'Modality'):
            logger.debug(f"Modality: {ds.Modality}")
        if hasattr(ds, 'SeriesDescription'):
            logger.debug(f"Series: {ds.SeriesDescription}")
        
        # Extract pixel array and convert to float32
        logger.debug("Converting pixel array to float32")
        pixel_array = ds.pixel_array.astype(np.float32)
        
        # Log image dimensions
        logger.debug(f"Image dimensions: {pixel_array.shape}")
        
        # Apply rescaling if attributes are present
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            slope = ds.RescaleSlope
            intercept = ds.RescaleIntercept
            logger.debug(f"Applying rescale: slope={slope}, intercept={intercept}")
            pixel_array = pixel_array * slope + intercept
        else:
            logger.debug("No rescale parameters found in DICOM header")
        
        # Log pixel value range
        logger.debug(f"Pixel value range: [{pixel_array.min():.2f}, {pixel_array.max():.2f}]")
        
        logger.info(f"Successfully loaded DICOM: {os.path.basename(file_path)}")
        return pixel_array, ds
        
    except Exception as e:
        logger.error(f"Error reading DICOM file {file_path}: {str(e)}")
        logger.exception("DICOM loading exception details:")
        raise IOError(f"Error reading DICOM file {file_path}: {e}")

