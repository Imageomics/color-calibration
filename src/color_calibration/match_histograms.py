import numpy as np
import cv2

def create_color_mapping(ref_hist, tgt_hist):
    """
    Create a mapping function that maps target intensity values to reference values,
    based on their normalized cumulative histograms.
    
    Args:
        ref_hist (np.ndarray): Normalized cumulative histogram (CDF) of the reference color card.
        tgt_hist (np.ndarray): Normalized cumulative histogram (CDF) of the target color card.
        
    Returns:
        np.ndarray: A lookup table mapping each intensity (0-255) from target to reference.
    """
    color_map = np.zeros(256, dtype=np.uint8)
    k = 0
    for src_i in range(256):
        while k < 255 and ref_hist[k] < tgt_hist[src_i]:
            k += 1
        color_map[src_i] = k
    return color_map

def match_histograms(ref_card, tgt_card, tgt_image):
    """
    Perform channel-wise histogram matching based on the normalized foreground intensities.
    No resizing is performed—the normalized cumulative histogram is computed on the raw,
    segmented color card images. Then, for each channel, the mapping is applied to both
    the full target image and the extracted (segmented) target color card.
    
    Args:
        ref_card (np.ndarray): Extracted reference color card (segmented foreground only).
        tgt_card (np.ndarray): Extracted target color card (segmented foreground only).
        tgt_image (np.ndarray): Full target image (from which the target card was extracted).
        
    Returns:
        tuple:
          calibrated_full_image (np.ndarray): The transformed full target image.
          calibrated_tgt_card (np.ndarray): The transformed target color card (segmented).
    """
    # Split the reference and target color cards into channels (assumed to be in BGR)
    ref_channels = cv2.split(ref_card)
    tgt_channels = cv2.split(tgt_card)
    
    if len(ref_channels) != len(tgt_channels):
        raise ValueError("Mismatch in the number of channels between reference and target color cards.")
    
    # Compute the normalized cumulative histograms (CDFs) for foreground pixels only
    ref_histograms = []
    tgt_histograms = []
    
    for ref_ch, tgt_ch in zip(ref_channels, tgt_channels):
        # Extract only foreground (nonzero) pixels
        ref_pixels = ref_ch[ref_ch > 0]
        tgt_pixels = tgt_ch[tgt_ch > 0]
        
        if len(ref_pixels) == 0 or len(tgt_pixels) == 0:
            raise ValueError("No valid foreground pixels found in one of the color cards.")
        
        # Compute a 256-bin histogram for the foreground pixels
        ref_hist, _ = np.histogram(ref_pixels, bins=256, range=(0, 256))
        tgt_hist, _ = np.histogram(tgt_pixels, bins=256, range=(0, 256))
        
        # Compute the normalized cumulative histogram (CDF)
        ref_cdf = ref_hist.cumsum() / float(ref_hist.sum())
        tgt_cdf = tgt_hist.cumsum() / float(tgt_hist.sum())
        
        ref_histograms.append(ref_cdf)
        tgt_histograms.append(tgt_cdf)
    
    # Build the mapping (lookup table) for each channel
    color_maps = [
        create_color_mapping(ref_histograms[i], tgt_histograms[i])
        for i in range(len(ref_channels))
    ]
    
    # Apply the mapping to the full target image, channel-wise
    full_tgt_channels = cv2.split(tgt_image)
    calibrated_full_channels = [
        cv2.LUT(full_tgt_channels[i], color_maps[i]) for i in range(len(full_tgt_channels))
    ]
    calibrated_full_image = cv2.merge(calibrated_full_channels)
    calibrated_full_image = cv2.convertScaleAbs(calibrated_full_image)
    
    # Also apply the mapping to the target segmented color card
    calibrated_tgt_channels = [
        cv2.LUT(tgt_channels[i], color_maps[i]) for i in range(len(tgt_channels))
    ]
    calibrated_tgt_card = cv2.merge(calibrated_tgt_channels)
    calibrated_tgt_card = cv2.convertScaleAbs(calibrated_tgt_card)
    
    return calibrated_full_image, calibrated_tgt_card
