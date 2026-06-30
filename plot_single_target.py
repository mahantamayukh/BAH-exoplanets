"""
plot_single_target.py
======================
Generates a fit_TIC{id}.png for any single TIC ID — useful for the
unlabeled survey candidates (e.g. TIC 300560295, TIC 197617428,
TIC 31508244) which were classified by survey_unlabeled.py but never
plotted, since that script only classifies and does not call plot_fit().

This version also runs classification and embeds the result (class +
confidence) into the plot title, and prints a clean summary of the
three required fitted parameters: orbital period, transit duration,
and transit depth.

Usage:
    python plot_single_target.py 300560295
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import lightkurve as lk
from parameter_fitter import fit_parameters, print_results, plot_fit
from feature_extractor import extract_features
from classifier import classify_lightcurve, load_classifier
from PIL import Image, ImageDraw, ImageFont


def clean_lightcurve(lc_raw):
    return lc_raw.remove_nans().remove_outliers(sigma_lower=10, sigma_upper=5)


def add_parameter_panel(image_path, tic_id, class_name, confidence, params):
    """
    Opens the saved fit plot and appends a parameter summary table to its
    RIGHT side only — output width increases, height stays identical to
    the original figure. Font size and panel width scale with the image's
    height so the table stays readable on high-resolution plots.
    """
    img = Image.open(image_path)
    orig_w, orig_h = img.size

    # Scale panel + fonts relative to image height (baseline: 700px -> sizes below)
    scale = orig_h / 700.0
    panel_w = int(420 * scale)
    size_title = max(int(22 * scale), 14)
    size_label = max(int(17 * scale), 11)
    size_value = max(int(17 * scale), 11)

    new_img = Image.new("RGB", (orig_w + panel_w, orig_h), "white")
    new_img.paste(img, (0, 0))

    draw = ImageDraw.Draw(new_img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size_title)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size_label)
        font_value = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size_value)
    except Exception:
        font_title = font_label = font_value = ImageFont.load_default()

    x0 = orig_w + int(28 * scale)
    y = int(36 * scale)
    panel_color = "#1a2744"
    line_gap_label = int(26 * scale)
    line_gap_value = int(24 * scale)
    section_gap = int(40 * scale)

    draw.text((x0, y), f"TIC {tic_id}", font=font_title, fill=panel_color)
    y += int(42 * scale)
    draw.line([(x0, y), (orig_w + panel_w - int(28 * scale), y)], fill="#cccccc", width=max(1, int(scale)))
    y += int(24 * scale)

    draw.text((x0, y), "Classification", font=font_label, fill=panel_color)
    y += line_gap_label
    draw.text((x0, y), f"{class_name}", font=font_value, fill="#222222")
    y += line_gap_value
    draw.text((x0, y), f"Confidence: {confidence*100:.1f}%", font=font_value, fill="#222222")
    y += section_gap

    rows = [
        ("Orbital Period", f"{params['period_med']:.5f} ± {params['period_err']:.5f} d"),
        ("Transit Duration", f"{params['duration_med']*24:.3f} ± {params['duration_err']*24:.3f} hr"),
        ("Transit Depth", f"{params['depth_med']*100:.4f} ± {params['depth_err']*100:.4f} %"),
        ("SNR", f"{params['snr']:.2f}"),
        ("Significance", f"{params['sigma']:.1f} σ"),
    ]
    for label, value in rows:
        draw.text((x0, y), label, font=font_label, fill=panel_color)
        y += line_gap_label
        draw.text((x0, y), value, font=font_value, fill="#222222")
        y += section_gap

    new_img.save(image_path)
    print(f"  ✓ Parameter panel appended (width increased, height unchanged)")


def main(tic_id):
    print(f"Downloading TIC {tic_id}...")
    search = lk.search_lightcurve(f'TIC {tic_id}', mission='TESS', exptime=120)
    if len(search) == 0:
        search = lk.search_lightcurve(f'TIC {tic_id}', mission='TESS')
    if len(search) == 0:
        print(f"✗ No data found for TIC {tic_id}")
        return

    lc_raw = search[0].download()
    lc_clean = clean_lightcurve(lc_raw)
    lc_flat = lc_clean.flatten(window_length=401, break_tolerance=5)

    # ── CLASSIFY ──────────────────────────────────────────────────────────────
    print("Loading classifier and classifying signal...")
    clf = load_classifier('classifier_model.pkl')
    result = classify_lightcurve(lc_flat, clf, tic_id=tic_id)
    print(f"  ✓ Classified as: {result['class_name']} "
          f"(confidence: {result['confidence']*100:.1f}%)")

    # ── FIT PARAMETERS ───────────────────────────────────────────────────────
    print("Running BLS + bootstrap fit...")
    params = fit_parameters(lc_flat, n_bootstrap=300)
    print_results(params, tic_id=tic_id, name=f"TIC {tic_id}")

    # ── CLEAN SUMMARY OF THE 3 REQUIRED PARAMETERS ──────────────────────────
    print(f"\n{'='*55}")
    print(f"  REQUIRED PARAMETER SUMMARY — TIC {tic_id}")
    print(f"{'='*55}")
    print(f"  Classification    : {result['class_name']} ({result['confidence']*100:.1f}% confidence)")
    print(f"  Orbital Period    : {params['period_med']:.5f} ± {params['period_err']:.5f} days")
    print(f"  Transit Duration  : {params['duration_med']*24:.3f} ± {params['duration_err']*24:.3f} hours")
    print(f"  Transit Depth     : {params['depth_med']*100:.4f} ± {params['depth_err']*100:.4f} %")
    print(f"  SNR / Significance: {params['snr']:.2f}  ({params['sigma']:.1f}σ)")
    print(f"{'='*55}")

    # ── PLOT, WITH CLASSIFICATION EMBEDDED IN TITLE ─────────────────────────
    plot_name = (f"TIC {tic_id}  —  Classified: {result['class_name']} "
                 f"({result['confidence']*100:.0f}% confidence)")
    save_path = f"fit_TIC{tic_id}.png"
    plot_fit(params, tic_id=tic_id, name=plot_name, save_path=save_path)

    add_parameter_panel(save_path, tic_id, result['class_name'], result['confidence'], params)

    print(f"\nSaved: {save_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_single_target.py <TIC_ID>")
        sys.exit(1)
    main(sys.argv[1])

