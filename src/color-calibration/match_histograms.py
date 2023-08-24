import numpy as np
import cv2

def create_color_mapping(ref, tgt):
    color_map = np.zeros(256)
    k = 0
    for src_i in range(len(tgt)):
        for ref_i in range(len(ref)):
            if ref[ref_i] >= tgt[src_i]:
                k = ref_i
                break
        color_map[src_i] = k
    return color_map

def match_histograms(ref_card, tgt_card, tgt_image):
    """
    Args:
        ref_card (cv:Mat): Reference color card
        tgt_card (cv:Mat): Target image color card
        tgt_image (cv:Mat): Target image to calibrate
    Description:
        Calibrates the target image given the color cards via the histogram matching algorithm.
        Histogram matching: https://en.wikipedia.org/wiki/Histogram_matching
        This algorithm creates a mapping between the reference color card and the target color card,
        then it applys the mapping to the whole target image. Colors not found in the color card images
        will be interpolated when being mapped.

    Assumptions:
        Assumes the color cards are in the same orientation
    """

    tgt_card = cv2.resize(tgt_card, ref_card.shape[:2], interpolation= cv2.INTER_LINEAR)

    ref_channels = list(cv2.split(ref_card))
    tgt_channels = list(cv2.split(tgt_card))
    assert len(ref_channels) == len(tgt_channels), "# of reference image channels must match # of target image channels"

    histograms = [ np.histogram(x.flatten(), 256, [0, 256])[0] for x in ref_channels + tgt_channels ]

    norm_histograms = [ x.cumsum() / float(x.cumsum().max()) for x in histograms ]

    color_maps = [create_color_mapping(norm_histograms[i], norm_histograms[i+len(ref_channels)]) for i in range(len(ref_channels))]

    full_tgt_channels = list(cv2.split(tgt_image))
    calibrated_channels = [cv2.LUT(full_tgt_channels[i], color_maps[i]) for i in range(len(full_tgt_channels))]

    calibrated_image = cv2.merge(calibrated_channels)
    calibrated_image = cv2.convertScaleAbs(calibrated_image)

    return calibrated_image