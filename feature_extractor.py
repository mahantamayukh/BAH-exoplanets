"""
feature_extractor.py
====================
Takes a cleaned light curve and BLS results, produces a feature
vector that the classifier can use to distinguish:
  - Planet transit
  - Eclipsing binary
  - Blend / false positive
  - Other / noise

Features extracted:
  1.  bls_power          — peak BLS power (how strong the periodic signal is)
  2.  snr                — signal-to-noise ratio of the dip
  3.  period             — best-fit period in days
  4.  duration           — transit duration in days
  5.  depth              — transit depth (fractional flux drop)
  6.  duty_cycle         — duration / period (fraction of orbit spent transiting)
  7.  depth_asymmetry    — odd vs even transit depth difference (EB signature)
  8.  secondary_depth    — depth at phase 0.5 (secondary eclipse = EB signature)
  9.  ingress_sharpness  — how sharp the transit edges are (U vs V shape)
  10. scatter_in         — flux scatter inside transit
  11. scatter_out        — flux scatter outside transit
  12. scatter_ratio      — scatter_in / scatter_out
  13. n_transits         — number of transits observed
  14. period_harmonic    — ratio of best period to its strongest harmonic
"""

import numpy as np
from astropy.timeseries import BoxLeastSquares
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(lc_flat, tic_id=None):
    """
    Given a flattened lightkurve LightCurve object, run BLS and extract
    all features needed for classification.

    Parameters
    ----------
    lc_flat : lightkurve.LightCurve
        Cleaned, flattened light curve (normalized around 1.0)
    tic_id : str or int, optional
        TIC ID for labeling purposes

    Returns
    -------
    features : dict
        Dictionary of all extracted features
    meta : dict
        BLS results and folded curve for plotting
    """

    time = np.asarray(lc_flat.time.value)
    flux = np.asarray(lc_flat.flux.value)

    # Remove any remaining NaNs/infs
    mask = np.isfinite(time) & np.isfinite(flux)
    time = time[mask]
    flux = flux[mask]

    # ── BLS PERIOD SEARCH ────────────────────────────────────────────────────
    periods   = np.linspace(0.5, 45.0, 15000)
    durations = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20])

    model   = BoxLeastSquares(time, flux)
    results = model.power(periods, durations)

    best_idx      = np.argmax(results.power)
    best_period   = float(results.period[best_idx])
    best_duration = float(results.duration[best_idx])
    best_depth    = float(results.depth[best_idx])
    best_t0       = float(results.transit_time[best_idx])
    best_power    = float(results.power[best_idx])

    # ── PHASE FOLD ───────────────────────────────────────────────────────────
    folded      = lc_flat.fold(period=best_period, epoch_time=best_t0)
    phase       = np.asarray(folded.phase.value)
    flux_folded = np.asarray(folded.flux.value)

    # Sort by phase for easier windowing
    sort_idx    = np.argsort(phase)
    phase       = phase[sort_idx]
    flux_folded = flux_folded[sort_idx]

    # Transit half-width in phase units
    half_dur = (best_duration / best_period) / 2.0

    in_mask  = np.abs(phase) <= half_dur
    out_mask = np.abs(phase) >  half_dur

    # ── FEATURE 1–2: SNR ─────────────────────────────────────────────────────
    out_mean    = np.mean(flux_folded[out_mask]) if out_mask.sum() > 0 else 1.0
    in_mean     = np.mean(flux_folded[in_mask])  if in_mask.sum()  > 0 else 1.0
    out_scatter = np.std(flux_folded[out_mask])  if out_mask.sum() > 0 else 1e-5
    out_scatter = max(out_scatter, 1e-9)          # avoid division by zero

    snr = (out_mean - in_mean) / out_scatter

    # ── FEATURE 3–6: BASIC TRANSIT PARAMS ───────────────────────────────────
    duty_cycle = best_duration / best_period

    # ── FEATURE 7: ODD-EVEN DEPTH ASYMMETRY (EB signature) ──────────────────
    # For EBs, odd and even eclipses have different depths (two stars of
    # different sizes). For planets, all transits are identical.
    time_span    = time[-1] - time[0]
    n_transits   = max(int(time_span / best_period), 1)

    odd_depths   = []
    even_depths  = []

    for i in range(n_transits):
        t_center = best_t0 + i * best_period
        half_d   = best_duration / 2.0
        tmask    = (time >= t_center - half_d) & (time <= t_center + half_d)

        if tmask.sum() < 3:
            continue

        transit_flux = flux[tmask]
        depth_i      = 1.0 - np.mean(transit_flux)

        if i % 2 == 0:
            even_depths.append(depth_i)
        else:
            odd_depths.append(depth_i)

    if len(odd_depths) > 0 and len(even_depths) > 0:
        odd_mean  = np.mean(odd_depths)
        even_mean = np.mean(even_depths)
        denom     = max(abs(odd_mean) + abs(even_mean), 1e-9)
        depth_asymmetry = abs(odd_mean - even_mean) / denom
    else:
        depth_asymmetry = 0.0

    # ── FEATURE 8: SECONDARY ECLIPSE (EB signature) ──────────────────────────
    # At phase ±0.5, a real EB will show a second dip (secondary eclipse).
    # Planets have NO secondary eclipse (or a tiny one from planet's thermal
    # emission, but negligible at TESS precision for most cases).
    sec_mask = np.abs(np.abs(phase) - 0.5) <= half_dur

    if sec_mask.sum() > 3:
        sec_mean      = np.mean(flux_folded[sec_mask])
        secondary_depth = max(out_mean - sec_mean, 0.0)
    else:
        secondary_depth = 0.0

    # Normalize secondary depth relative to primary depth
    primary_depth_measured = max(out_mean - in_mean, 1e-9)
    secondary_ratio        = secondary_depth / primary_depth_measured

    # ── FEATURE 9: INGRESS SHARPNESS (U vs V shape) ──────────────────────────
    # U-shaped = flat bottom = planet (limb darkening + full ingress/egress)
    # V-shaped = pointed bottom = grazing EB or blend
    # Measure: variance of flux at bottom of transit vs edges
    if in_mask.sum() > 5:
        bottom_10pct = phase[in_mask][
            np.abs(phase[in_mask]) < half_dur * 0.3
        ]
        edge_mask = (np.abs(phase) > half_dur * 0.5) & (np.abs(phase) < half_dur)

        if len(bottom_10pct) > 2 and edge_mask.sum() > 2:
            bottom_flux    = flux_folded[np.isin(phase, bottom_10pct)]
            edge_flux      = flux_folded[edge_mask]
            bottom_scatter = np.std(bottom_flux)
            edge_scatter   = np.std(edge_flux)
            ingress_sharpness = edge_scatter / max(bottom_scatter, 1e-9)
        else:
            ingress_sharpness = 1.0
    else:
        ingress_sharpness = 1.0

    ingress_sharpness = min(ingress_sharpness, 10.0)  # cap outliers

    # ── FEATURE 10–12: SCATTER RATIO ─────────────────────────────────────────
    scatter_in    = np.std(flux_folded[in_mask])  if in_mask.sum()  > 2 else out_scatter
    scatter_out   = out_scatter
    scatter_ratio = scatter_in / max(scatter_out, 1e-9)
    scatter_ratio = min(scatter_ratio, 10.0)

    # ── FEATURE 14: PERIOD HARMONIC ──────────────────────────────────────────
    # Check if the true period might be 2x the detected period
    # (BLS sometimes finds half the period for EBs with equal-depth eclipses)
    harmonic_period = best_period * 2.0
    if harmonic_period <= 20.0:
        results_harm  = model.power(
            np.array([harmonic_period]),
            durations
        )
        harmonic_power = float(results_harm.power[0])
        period_harmonic = harmonic_power / max(best_power, 1e-9)
    else:
        period_harmonic = 0.0

    # ── ASSEMBLE FEATURE DICT ────────────────────────────────────────────────
    features = {
        'tic_id'             : str(tic_id) if tic_id else 'unknown',
        'bls_power'          : best_power,
        'snr'                : snr,
        'period'             : best_period,
        'duration'           : best_duration,
        'depth'              : best_depth,
        'duty_cycle'         : duty_cycle,
        'depth_asymmetry'    : depth_asymmetry,
        'secondary_ratio'    : secondary_ratio,
        'ingress_sharpness'  : ingress_sharpness,
        'scatter_in'         : scatter_in,
        'scatter_out'        : scatter_out,
        'scatter_ratio'      : scatter_ratio,
        'n_transits'         : n_transits,
        'period_harmonic'    : period_harmonic,
    }

    meta = {
        'best_period'   : best_period,
        'best_duration' : best_duration,
        'best_depth'    : best_depth,
        'best_t0'       : best_t0,
        'best_power'    : best_power,
        'bls_results'   : results,
        'phase'         : phase,
        'flux_folded'   : flux_folded,
        'in_mask'       : in_mask,
        'out_mask'      : out_mask,
        'snr'           : snr,
    }

    return features, meta


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE NAMES (for XGBoost — excludes tic_id)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    'bls_power',
    'snr',
    'period',
    'duration',
    'depth',
    'duty_cycle',
    'depth_asymmetry',
    'secondary_ratio',
    'ingress_sharpness',
    'scatter_in',
    'scatter_out',
    'scatter_ratio',
    'n_transits',
    'period_harmonic',
]


def features_to_vector(features):
    """Convert features dict to numpy array for classifier input."""
    return np.array([features[k] for k in FEATURE_NAMES], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import lightkurve as lk

    print("Testing feature extractor on WASP-18 (TIC 100100827)...")
    lc_raw  = lk.search_lightcurve('TIC 100100827', mission='TESS', exptime=120)[0].download()
    lc_clean = lc_raw.remove_nans().remove_outliers(sigma_lower=10, sigma_upper=5)
    lc_flat  = lc_clean.flatten(window_length=401, break_tolerance=5)

    features, meta = extract_features(lc_flat, tic_id=100100827)

    print("\n── Extracted Features ──────────────────────────")
    for k, v in features.items():
        if k == 'tic_id':
            print(f"  {k:<22}: {v}")
        else:
            print(f"  {k:<22}: {v:.6f}")

    print(f"\n── Interpretation ──────────────────────────────")
    print(f"  SNR {meta['snr']:.1f}  → {'Detection ✅' if meta['snr'] > 7 else 'Weak signal ⚠️'}")
    print(f"  Secondary ratio {features['secondary_ratio']:.3f} → {'Possible EB ⚠️' if features['secondary_ratio'] > 0.3 else 'Planet-like ✅'}")
    print(f"  Depth asymmetry {features['depth_asymmetry']:.3f} → {'Possible EB ⚠️' if features['depth_asymmetry'] > 0.1 else 'Planet-like ✅'}")
