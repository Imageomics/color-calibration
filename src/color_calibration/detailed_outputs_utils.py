import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import cv2
import base64
from io import BytesIO
from PIL import Image

def cv2_to_base64(img):
    """Convert a cv2 image (BGR) to a base64-encoded PNG image string."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + encoded

def save_histogram_detailed_outputs(ref_card, orig_tgt_card, calibrated_tgt_card, output_path):
    """
    Save an interactive Plotly detailed outputs figure as an HTML file that overlays per-channel histograms for:
      - the reference color card,
      - the original target color card, and
      - the transformed target color card.
    Inset images of each card are added.
    """
    bins = np.arange(257)
    x_values = bins[:-1]
    channel_colors = {0: 'blue', 1: 'green', 2: 'red'}
    channel_names = {0: 'Blue', 1: 'Green', 2: 'Red'}
    line_styles = {
        'Reference': 'solid',
        'Target Original': 'dash',
        'Target Transformed': 'dot'
    }
    fig = go.Figure()
    for i in range(3):
        # Extract only foreground (nonzero) pixels
        ref_pixels = ref_card[:, :, i][ref_card[:, :, i] > 0].flatten()
        orig_pixels = orig_tgt_card[:, :, i][orig_tgt_card[:, :, i] > 0].flatten()
        trans_pixels = calibrated_tgt_card[:, :, i][calibrated_tgt_card[:, :, i] > 0].flatten()
        
        hist_ref, _ = np.histogram(ref_pixels, bins=bins, range=(0,256))
        hist_orig, _ = np.histogram(orig_pixels, bins=bins, range=(0,256))
        hist_trans, _ = np.histogram(trans_pixels, bins=bins, range=(0,256))
        if hist_ref.sum() > 0:
            hist_ref = hist_ref / hist_ref.sum()
        if hist_orig.sum() > 0:
            hist_orig = hist_orig / hist_orig.sum()
        if hist_trans.sum() > 0:
            hist_trans = hist_trans / hist_trans.sum()
        fig.add_trace(go.Scatter(
            x=x_values,
            y=hist_ref,
            mode='lines',
            line=dict(color=channel_colors[i], dash=line_styles['Reference']),
            name=f"Reference {channel_names[i]}"
        ))
        fig.add_trace(go.Scatter(
            x=x_values,
            y=hist_orig,
            mode='lines',
            line=dict(color=channel_colors[i], dash=line_styles['Target Original']),
            name=f"Target Original {channel_names[i]}"
        ))
        fig.add_trace(go.Scatter(
            x=x_values,
            y=hist_trans,
            mode='lines',
            line=dict(color=channel_colors[i], dash=line_styles['Target Transformed']),
            name=f"Target Transformed {channel_names[i]}"
        ))
    ref_img_url = cv2_to_base64(ref_card)
    orig_tgt_url = cv2_to_base64(orig_tgt_card)
    trans_tgt_url = cv2_to_base64(calibrated_tgt_card)
    fig.add_layout_image(
        dict(
            source=ref_img_url,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            sizex=0.15, sizey=0.15,
            xanchor="left", yanchor="top",
            opacity=0.9,
            layer="above"
        )
    )
    fig.add_layout_image(
        dict(
            source=orig_tgt_url,
            xref="paper", yref="paper",
            x=0.02, y=0.80,
            sizex=0.15, sizey=0.15,
            xanchor="left", yanchor="top",
            opacity=0.9,
            layer="above"
        )
    )
    fig.add_layout_image(
        dict(
            source=trans_tgt_url,
            xref="paper", yref="paper",
            x=0.02, y=0.62,
            sizex=0.15, sizey=0.15,
            xanchor="left", yanchor="top",
            opacity=0.9,
            layer="above"
        )
    )
    fig.add_annotation(dict(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text="Reference",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color="black", size=12)
    ))
    fig.add_annotation(dict(
        x=0.02, y=0.80,
        xref="paper", yref="paper",
        text="Target Original",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color="black", size=12)
    ))
    fig.add_annotation(dict(
        x=0.02, y=0.62,
        xref="paper", yref="paper",
        text="Target Transformed",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color="black", size=12)
    ))
    fig.update_layout(
        title="Overlayed Histograms by Channel",
        xaxis_title="Intensity",
        yaxis_title="Normalized Frequency",
        legend_title="Histogram Source",
        template="plotly_white"
    )
    out_path = Path(output_path)
    if out_path.suffix.lower() != '.html':
        out_path = out_path.with_suffix('.html')
    fig.write_html(str(out_path))

def save_color_card_visualizations(image, segmentation, output_path):
    """
    A wrapper function to generate color card extraction detailed outputs visualizations.
    This function calls the existing 'visualize_color_card_extraction' from coco_utils.
    """
    from .coco_utils import visualize_color_card_extraction
    # We assume that the detailed outputs directory is managed by the caller.
    return visualize_color_card_extraction(image, segmentation, output_path)

def save_all_detailed_outputs(ref_image, ref_seg, ref_card,
                           tgt_image, tgt_seg, tgt_card, tgt_mask,
                           calibrated_full_image, calibrated_tgt_card,
                           detailed_outputs_dir, base_filename):
    """
    Save all detailed outputs for a single calibration operation.
    The following are saved:
      - The reference segmented color card (ref_card)
      - The reference full image (ref_image)
      - The original target segmented color card (tgt_card)
      - The transformed target segmented color card (calibrated_tgt_card)
      - The original target full image (tgt_image)
      - The transformed target full image (calibrated_full_image)
    """
    detailed_outputs_dir = Path(detailed_outputs_dir) / "detailed-outputs"
    detailed_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save reference segmented color card
    ref_card_path = detailed_outputs_dir / f"ref_color_card_{base_filename}.png"
    cv2.imwrite(str(ref_card_path), ref_card)
    
    # Save reference full image
    ref_full_path = detailed_outputs_dir / f"ref_full_image_{base_filename}.png"
    cv2.imwrite(str(ref_full_path), ref_image)
    
    # Save original target segmented color card
    tgt_card_path = detailed_outputs_dir / f"orig_tgt_color_card_{base_filename}.png"
    cv2.imwrite(str(tgt_card_path), tgt_card)
    
    # Save transformed target segmented color card
    trans_tgt_card_path = detailed_outputs_dir / f"trans_tgt_color_card_{base_filename}.png"
    cv2.imwrite(str(trans_tgt_card_path), calibrated_tgt_card)
    
    # Save original target full image
    tgt_full_path = detailed_outputs_dir / f"orig_tgt_full_image_{base_filename}.png"
    cv2.imwrite(str(tgt_full_path), tgt_image)
    
    # Save transformed target full image
    trans_tgt_full_path = detailed_outputs_dir / f"trans_tgt_full_image_{base_filename}.png"
    cv2.imwrite(str(trans_tgt_full_path), calibrated_full_image)
    
    print(f"Saved detailed outputs images to {detailed_outputs_dir}")
