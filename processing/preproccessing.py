import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, ball

# Pre-allocate the standard mask at module level for reuse
_standard_mask = np.zeros((64, 128, 128), dtype=bool)
_standard_mask[20:40, 82:103, 43:82] = 1

# Pre-compute mask region indices for direct access
_MASK_REGION = (slice(20, 40), slice(82, 103), slice(43, 82))

# Pre-compute resize parameters for the standard case (64, 128, 128)
# These are the most common input dimensions we expect based on dataloader.py
_STD_INPUT_SHAPE = (64, 128, 128)  # From volume[9:73, :, :]

# Common case 1: Input is (64, h, w) where h < 128 and w < 128 (padding needed)
_STD_H_PADDING = {h: (128 - h) // 2 for h in range(64, 129)}  # Common heights
_STD_W_PADDING = {w: (128 - w) // 2 for w in range(64, 129)}  # Common widths

# Common case 2: Input is (64, h, w) where h > 128 and w > 128 (cropping needed)
_STD_H_CROPPING = {h: (h - 128) // 2 for h in range(129, 257)}  # Common heights
_STD_W_CROPPING = {w: (w - 128) // 2 for w in range(129, 257)}  # Common widths

def resize_volume(volume, target_shape=(64, 128, 128)):
    """
    Optimized version that resizes the volume to the target shape (64, 128, 128) 
    using zero-padding or center cropping. This version is optimized for the 
    specific case where we know all images have the same target dimensions.

    Args:
        volume: Input 3D volume as numpy array with shape (d, h, w)
        target_shape: Desired output shape as tuple (d_new, h_new, w_new)

    Returns:
        Resized volume with shape target_shape
    """
    # Fast path for the common case: input from volume[9:73, :, :] to (64, 128, 128)
    # This is the case we see in the dataloader
    if volume.shape[0] == 64 and target_shape == (64, 128, 128):
        return _resize_volume_standard_case(volume)
    
    # General case for other dimensions
    d, h, w = volume.shape
    target_d, target_h, target_w = target_shape
    
    # Create output array of target size
    resized = np.zeros(target_shape, dtype=volume.dtype)
    
    # Calculate actual dimensions to copy (min of source and target)
    copy_d = min(d, target_d)
    copy_h = min(h, target_h)
    copy_w = min(w, target_w)
    
    # Calculate start indices for source (for cropping)
    src_d_start = (d - copy_d) // 2 if d > target_d else 0
    src_h_start = (h - copy_h) // 2 if h > target_h else 0
    src_w_start = (w - copy_w) // 2 if w > target_w else 0
    
    # Calculate start indices for target (for padding)
    tgt_d_start = (target_d - copy_d) // 2 if target_d > d else 0
    tgt_h_start = (target_h - copy_h) // 2 if target_h > h else 0
    tgt_w_start = (target_w - copy_w) // 2 if target_w > w else 0
    
    # Copy the appropriate slice from source to target
    resized[tgt_d_start:tgt_d_start+copy_d, 
            tgt_h_start:tgt_h_start+copy_h, 
            tgt_w_start:tgt_w_start+copy_w] = \
    volume[src_d_start:src_d_start+copy_d, 
           src_h_start:src_h_start+copy_h, 
           src_w_start:src_w_start+copy_w]
    
    return resized


def _resize_volume_standard_case(volume):
    """
    Highly optimized version for the specific case where:
    - Input is from a slice with depth 64 (like volume[9:73, :, :])
    - Target shape is (64, 128, 128)
    
    This uses pre-computed parameters and NumPy's vectorized operations for maximum performance.
    """
    h, w = volume.shape[1], volume.shape[2]
    
    # Create the output array of the target size - reuse the same shape
    resized = np.zeros(_STD_INPUT_SHAPE, dtype=volume.dtype)
    
    # Handle both height and width using pre-computed parameters
    if h <= 128 and w <= 128:
        # Both dimensions need padding - use lookup tables
        h_start = _STD_H_PADDING.get(h, (128 - h) // 2)  # Fallback if not in table
        w_start = _STD_W_PADDING.get(w, (128 - w) // 2)  # Fallback if not in table
        resized[:, h_start:h_start+h, w_start:w_start+w] = volume
    
    elif h <= 128 and w > 128:
        # Height needs padding, width needs cropping
        h_start = _STD_H_PADDING.get(h, (128 - h) // 2)
        w_start = _STD_W_CROPPING.get(w, (w - 128) // 2)
        resized[:, h_start:h_start+h, :] = volume[:, :, w_start:w_start+128]
    
    elif h > 128 and w <= 128:
        # Height needs cropping, width needs padding
        h_start = _STD_H_CROPPING.get(h, (h - 128) // 2)
        w_start = _STD_W_PADDING.get(w, (128 - w) // 2)
        resized[:, :, w_start:w_start+w] = volume[:, h_start:h_start+128, :]
    
    else:  # h > 128 and w > 128
        # Both dimensions need cropping
        h_start = _STD_H_CROPPING.get(h, (h - 128) // 2)
        w_start = _STD_W_CROPPING.get(w, (w - 128) // 2)
        resized = volume[:, h_start:h_start+128, w_start:w_start+128]
    
    return resized

def process_volume(volume, target_shape=(64, 128, 128)):
    """
    Process a 3D volume by:
    1. Normalizing intensity by subtracting minimum
    2. Resizing to target_shape
    3. Applying a hard-coded brain mask and normalizing by the mean value in the mask

    Args:
        volume: Input 3D volume
        target_shape: Desired output shape (depth, height, width)

    Returns:
        norm_vol: Normalized and resized volume
        mask: Brain mask
        masked_vol: Masked volume (None in this implementation)
    """
    # Fast path for the common case from dataloader: volume[9:73, :, :]
    if volume.shape[0] == 64 and target_shape == (64, 128, 128):
        return _process_volume_standard_case(volume)
    
    # General case
    # 1. Normalize by subtracting minimum
    norm_vol = volume - volume.min()
    
    # 2. Resize the normalized volume
    norm_vol = resize_volume(norm_vol, target_shape=target_shape)
    
    # 3. Apply the hard-coded brain mask
    mask = np.zeros(target_shape, dtype=bool)
    mask[20:40, 82:103, 43:82] = 1
    
    # 4. Normalize by mean value in mask region
    mean_val = np.mean(norm_vol[mask])
    if mean_val > 1e-10:  # Avoid division by very small numbers
        norm_vol /= mean_val
    else:
        print(f"WARNING: Very small mean value in mask region: {mean_val}")

    return norm_vol, mask, None


def _process_volume_standard_case(volume):
    """
    Highly optimized version of process_volume for the standard case with:
    - Input volume with shape (64, h, w)
    - Target shape (64, 128, 128)
    - Hard-coded mask at mask[20:40, 82:103, 43:82]
    
    This uses pre-computed parameters and avoids redundant calculations.
    """
    # 1. Normalize by subtracting minimum (vectorized)
    min_val = volume.min()
    norm_vol = volume - min_val
    
    # 2. Resize using the optimized function with pre-computed parameters
    norm_vol = _resize_volume_standard_case(norm_vol)
    
    # 3. Use the pre-defined module-level mask region
    # Direct access to the mask region for mean calculation using pre-computed slices
    masked_values = norm_vol[_MASK_REGION]
    mean_val = np.mean(masked_values)
    
    if mean_val > 1e-10:  # Avoid division by very small numbers
        norm_vol /= mean_val
    else:
        print(f"WARNING: Very small mean value in mask region: {mean_val}")

    return norm_vol, _standard_mask, None
