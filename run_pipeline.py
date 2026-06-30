"""
run_pipeline.py
===============
Master pipeline — runs the full stack on all confirmed test targets:

  Download → Clean → Extract Features → Classify → Fit Parameters → Visualize

Produces:
  - results_summary.csv     : all targets with classification + parameters
  - pipeline_results.png    : multi-panel summary figure
  - Per-target fit plots     : fit_TIC{id}.png
  - confusion_matrix.png    : classifier evaluation
  - feature_importances.png : what features matter most

Run this AFTER training the classifier (classifier.py must have run once).
If no trained model exists, it trains one automatically.
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

import lightkurve as lk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from feature_extractor  import extract_features, features_to_vector
from classifier         import (collect_training_data, train_classifier,
                                 classify_lightcurve, load_classifier,
                                 CONFIRMED_PLANETS, CLASS_NAMES, CLASS_COLORS)
from parameter_fitter   import fit_parameters, print_results, plot_fit

# ─────────────────────────────────────────────────────────────────────────────
# TARGETS TO RUN
# ─────────────────────────────────────────────────────────────────────────────

TEST_TARGETS = CONFIRMED_PLANETS   # Use all 10 confirmed planets as test set


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE FOR ONE TARGET
# ─────────────────────────────────────────────────────────────────────────────

def run_one_target(target, clf):
    """Full pipeline for a single target. Returns result dict or None."""
    tic_id = target['tic_id']
    name   = target['name']

    print(f"\n{'━'*55}")
    print(f"  TARGET: {name} (TIC {tic_id})")
    print(f"{'━'*55}")

    # ── DOWNLOAD ─────────────────────────────────────────────────────────────
    try:
        print("  [1/4] Downloading light curve...", flush=True)
        search = lk.search_lightcurve(f'TIC {tic_id}', mission='TESS', exptime=120)
        if len(search) == 0:
            search = lk.search_lightcurve(f'TIC {tic_id}', mission='TESS')
        if len(search) == 0:
            print(f"  ✗ No data found for TIC {tic_id}")
            return None

        lc_raw = search[0].download()
        print(f"  ✓ {len(lc_raw)} data points downloaded")

    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return None

    # ── CLEAN ─────────────────────────────────────────────────────────────────
    print("  [2/4] Cleaning light curve...", flush=True)
    lc_clean = lc_raw.remove_nans().remove_outliers(sigma_lower=10, sigma_upper=5)
    lc_flat  = lc_clean.flatten(window_length=401, break_tolerance=5)
    print(f"  ✓ {len(lc_flat)} points after cleaning")

    # ── CLASSIFY ──────────────────────────────────────────────────────────────
    print("  [3/4] Classifying signal...", flush=True)
    result = classify_lightcurve(lc_flat, clf, tic_id=tic_id)
    print(f"  ✓ Classified as: {result['class_name']} "
          f"(confidence: {result['confidence']*100:.1f}%)")
    print(f"  ✓ SNR: {result['snr']:.1f}")

    # ── FIT PARAMETERS ────────────────────────────────────────────────────────
    print("  [4/4] Fitting parameters with bootstrap...", flush=True)
    params = fit_parameters(lc_flat, n_bootstrap=300)
    print_results(params, tic_id=tic_id, name=name)

    # ── SAVE INDIVIDUAL PLOT ──────────────────────────────────────────────────
    fit_plot_path = f"fit_TIC{tic_id}.png"
    plot_fit(params, tic_id=tic_id, name=name, save_path=fit_plot_path)

    # ── CORRECTNESS CHECK ─────────────────────────────────────────────────────
    expected_period = target['period']
    period_ok       = abs(params['period_med'] - expected_period) < (0.1 * expected_period)

    return {
        'tic_id'          : tic_id,
        'name'            : name,
        'class_name'      : result['class_name'],
        'class_id'        : result['class_id'],
        'confidence'      : result['confidence'],
        'snr'             : result['snr'],
        'period_fit'      : params['period_med'],
        'period_err'      : params['period_err'],
        'period_known'    : expected_period,
        'period_ok'       : period_ok,
        'depth_fit'       : params['depth_med'],
        'depth_err'       : params['depth_err'],
        'depth_known'     : target['depth'],
        'duration_fit'    : params['duration_med'],
        'duration_err'    : params['duration_err'],
        'sigma'           : params['sigma'],
        'bls_power'       : params['bls_power'],
        'params'          : params,
        'result'          : result,
        'lc_flat'         : lc_flat,
        'lc_raw'          : lc_raw,
        'fit_plot'        : fit_plot_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def make_summary_figure(all_results):
    """
    Multi-panel figure showing phase-folded light curves for all targets
    with their classification and parameters.
    """
    valid   = [r for r in all_results if r is not None]
    n       = len(valid)
    n_cols  = 3
    n_rows  = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6 * n_cols, 4 * n_rows))
    fig.suptitle('Exoplanet Transit Pipeline — All Targets', fontsize=15, y=1.01)

    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, res in enumerate(valid):
        ax     = axes_flat[i]
        params = res['params']
        phase  = params['phase']
        flux   = params['flux']

        # Bin for clarity
        n_bins  = 60
        bins    = np.linspace(-0.5, 0.5, n_bins + 1)
        bin_mid = (bins[:-1] + bins[1:]) / 2
        bin_flux = []
        for j in range(n_bins):
            m = (phase >= bins[j]) & (phase < bins[j+1])
            bin_flux.append(np.mean(flux[m]) if m.sum() > 0 else np.nan)
        bin_flux = np.array(bin_flux)

        ax.scatter(phase, flux, s=0.5, alpha=0.15, color='steelblue')
        valid_b = np.isfinite(bin_flux)
        ax.plot(bin_mid[valid_b], bin_flux[valid_b], 'o-',
                ms=3, lw=1.2, color='navy', zorder=5)
        ax.axhline(1.0, color='gray', lw=0.7, linestyle='--')

        # Color-code by classification
        cls_id    = res['class_id']
        cls_color = CLASS_COLORS[cls_id]
        cls_name  = res['class_name']

        # Confidence badge
        conf_str = f"{res['confidence']*100:.0f}%"
        ax.set_facecolor('#f8f9fa')
        ax.patch.set_alpha(0.3)

        title = (f"{res['name']}\n"
                 f"P={params['period_med']:.3f}±{params['period_err']:.3f}d  "
                 f"SNR={params['snr']:.1f}")
        ax.set_title(title, fontsize=8)

        xlabel = (f"[{cls_name} — {conf_str} confidence]")
        ax.set_xlabel(xlabel, fontsize=7.5, color=cls_color, fontweight='bold')
        ax.set_ylabel('Normalized Flux', fontsize=8)
        ax.set_xlim(-0.5, 0.5)
        ax.tick_params(labelsize=7)

    # Hide empty panels
    for j in range(len(valid), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('pipeline_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: pipeline_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def save_results_csv(all_results):
    """Save all results to a clean CSV."""
    rows = []
    for r in all_results:
        if r is None:
            continue
        rows.append({
            'TIC ID'              : r['tic_id'],
            'Name'                : r['name'],
            'Classification'      : r['class_name'],
            'Confidence (%)'      : round(r['confidence'] * 100, 1),
            'SNR'                 : round(r['snr'], 1),
            'Period Fit (days)'   : round(r['period_fit'], 5),
            'Period Err (days)'   : round(r['period_err'], 5),
            'Period Known (days)' : r['period_known'],
            'Period Match'        : '✓' if r['period_ok'] else '✗',
            'Depth Fit (%)'       : round(r['depth_fit'] * 100, 4),
            'Depth Err (%)'       : round(r['depth_err'] * 100, 4),
            'Depth Known (%)'     : round(r['depth_known'] * 100, 3),
            'Duration Fit (hr)'   : round(r['duration_fit'] * 24, 3),
            'Duration Err (hr)'   : round(r['duration_err'] * 24, 3),
            'Detection Sigma'     : round(r['sigma'], 1),
        })

    df = pd.DataFrame(rows)
    df.to_csv('results_summary.csv', index=False)
    print("Saved: results_summary.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("EXOPLANET TRANSIT DETECTION PIPELINE")
    print("=" * 60)

    # ── STEP 1: LOAD OR TRAIN CLASSIFIER ─────────────────────────────────────
    model_file = 'classifier_model.pkl'

    if os.path.exists(model_file):
        print(f"\n[*] Loading existing classifier from {model_file}...")
        clf = load_classifier(model_file)
    else:
        print("\n[*] No classifier found — training now...")
        data = collect_training_data(cache_file='training_data.pkl')
        clf  = train_classifier(data, model_file=model_file)

    # ── STEP 2: RUN PIPELINE ON ALL TARGETS ──────────────────────────────────
    print(f"\n[*] Running pipeline on {len(TEST_TARGETS)} targets...")
    all_results = []

    for target in TEST_TARGETS:
        result = run_one_target(target, clf)
        all_results.append(result)

    # ── STEP 3: SUMMARY ───────────────────────────────────────────────────────
    valid = [r for r in all_results if r is not None]
    print(f"\n\n{'═'*60}")
    print(f"  PIPELINE COMPLETE — {len(valid)}/{len(TEST_TARGETS)} targets processed")
    print(f"{'═'*60}")

    n_correct_class = sum(1 for r in valid if r['class_name'] == 'Planet Transit')
    n_period_match  = sum(1 for r in valid if r['period_ok'])

    print(f"\n  Classification accuracy : {n_correct_class}/{len(valid)} "
          f"correctly classified as Planet Transit")
    print(f"  Period recovery         : {n_period_match}/{len(valid)} "
          f"within 10% of known period")

    # ── STEP 4: OUTPUTS ───────────────────────────────────────────────────────
    print("\n[*] Generating outputs...")
    make_summary_figure(all_results)
    df = save_results_csv(all_results)

    print(f"\n{'─'*60}")
    print("  OUTPUT FILES GENERATED:")
    print("  • pipeline_results.png    — phase-folded curves, all targets")
    print("  • results_summary.csv     — all parameters + classifications")
    print("  • confusion_matrix.png    — classifier evaluation")
    print("  • feature_importances.png — what features drive decisions")
    print("  • fit_TIC{id}.png         — individual parameter fit plots")
    print(f"{'─'*60}")

    # Print table
    print("\n  RESULTS TABLE:")
    print(df[['Name', 'Classification', 'Confidence (%)',
              'SNR', 'Period Fit (days)', 'Period Known (days)',
              'Period Match']].to_string(index=False))


if __name__ == '__main__':
    main()
