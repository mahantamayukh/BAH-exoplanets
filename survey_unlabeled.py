"""
survey_unlabeled.py
====================
Runs the already-trained classifier across a batch of UNLABELED TESS
light curve files (e.g. a random sample from a sector bulk download).

This is the "apply the classifier on the given science datasets and
correctly categorize the type of signals present" deliverable from
the project brief — distinct from run_pipeline.py, which only
re-validates against 10 already-known confirmed planets.

Usage:
    python survey_unlabeled.py

Expects:
    - classifier_model.pkl already exists (run run_pipeline.py once first)
    - A folder of .fits files, default: batch_lightcurves/

Produces:
    - sector_survey_results.csv   : one row per star, classification + params
    - survey_summary.png          : bar chart of class counts
"""

import glob
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightkurve as lk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from feature_extractor import extract_features
from classifier import classify_lightcurve, load_classifier, CLASS_COLORS, CLASS_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BATCH_DIR   = "batch_lightcurves"
MODEL_FILE  = "classifier_model.pkl"
OUTPUT_CSV  = "sector_survey_results.csv"
OUTPUT_PNG  = "survey_summary.png"


def clean_lightcurve(lc_raw):
    """Shared cleaning step — asymmetric sigma clip so real transit
    dips aren't clipped out, only upward spikes (momentum dumps etc)."""
    return lc_raw.remove_nans().remove_outliers(sigma_lower=10, sigma_upper=5)


def run_survey():
    if not os.path.exists(MODEL_FILE):
        print(f"✗ {MODEL_FILE} not found. Run run_pipeline.py first to train the classifier.")
        return

    clf = load_classifier(MODEL_FILE)

    fits_files = sorted(glob.glob(os.path.join(BATCH_DIR, "*.fits")))
    print(f"Found {len(fits_files)} FITS files in {BATCH_DIR}/")

    if len(fits_files) == 0:
        print(f"✗ No .fits files found in {BATCH_DIR}/ — check the download step.")
        return

    results = []
    n_failed = 0

    for i, f in enumerate(fits_files):
        fname = os.path.basename(f)
        print(f"[{i+1}/{len(fits_files)}] {fname}...", end=" ", flush=True)

        try:
            lc_raw = lk.read(f)
            lc_clean = clean_lightcurve(lc_raw)

            if len(lc_clean) < 100:
                print("skipped (too few points after cleaning)")
                n_failed += 1
                continue

            lc_flat = lc_clean.flatten(window_length=401, break_tolerance=5)

            tic_id = lc_raw.meta.get('TICID', fname)
            result = classify_lightcurve(lc_flat, clf, tic_id=tic_id)

            results.append({
                'tic_id'      : tic_id,
                'file'        : fname,
                'class'       : result['class_name'],
                'confidence'  : round(result['confidence'] * 100, 1),
                'snr'         : round(result['snr'], 2),
                'period'      : round(result['period'], 5) if result.get('period') else None,
                'depth_pct'   : round(result['depth'] * 100, 4) if result.get('depth') else None,
            })
            print(f"{result['class_name']} ({result['confidence']*100:.0f}%)")

        except Exception as e:
            print(f"FAILED ({e})")
            n_failed += 1
            continue

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*55}")
    print(f"  SURVEY COMPLETE")
    print(f"{'='*55}")
    print(f"  Processed successfully : {len(df)}/{len(fits_files)}")
    print(f"  Failed/skipped         : {n_failed}/{len(fits_files)}")
    print(f"\n  Classification breakdown:")
    print(df['class'].value_counts().to_string())
    print(f"\nSaved: {OUTPUT_CSV}")

    # ── Summary bar chart ────────────────────────────────────────────────────
    if len(df) > 0:
        counts = df['class'].value_counts()
        name_to_color = dict(zip(CLASS_NAMES, CLASS_COLORS))
        colors = [name_to_color.get(name, '#888888') for name in counts.index]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(counts.index, counts.values, color=colors)
        ax.set_ylabel('Number of stars')
        ax.set_title(f'Classification breakdown — {len(df)} stars surveyed\n'
                      f'(random sample, Sector 1)', fontsize=11)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     str(val), ha='center', fontsize=10)
        plt.tight_layout()
        plt.savefig(OUTPUT_PNG, dpi=150)
        plt.close()
        print(f"Saved: {OUTPUT_PNG}")

        # Flag interesting candidates worth showing individually in the report
        top_candidates = df[df['class'] == 'Planet Transit'].sort_values(
            'confidence', ascending=False).head(3)
        if len(top_candidates) > 0:
            print(f"\n  Top 'Planet Transit' candidates (for individual plots in report):")
            print(top_candidates.to_string(index=False))


if __name__ == '__main__':
    run_survey()
