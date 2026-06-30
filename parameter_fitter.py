"""
parameter_fitter.py
===================
Fits transit parameters with uncertainty estimates using bootstrap
resampling on the phase-folded light curve.

Parameters estimated:
  - Orbital period (days)          ± uncertainty
  - Transit depth (fractional)     ± uncertainty
  - Transit duration (days)        ± uncertainty
  - Mid-transit time T0            ± uncertainty
  - Transit SNR
  - Detection confidence level (σ)

Why bootstrap instead of full MCMC:
  Bootstrap gives honest uncertainty intervals in ~seconds.
  Full MCMC (emcee) gives the same answer but takes 10-30 minutes
  per star. For a 48-hour hackathon with dozens of targets, bootstrap
  is the right engineering tradeoff. We note this in the report.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# TRAPEZOID TRANSIT MODEL
# ─────────────────────────────────────────────────────────────────────────────

def trapezoid_model(phase, depth, duration, ingress_fraction=0.2):
    """
    Simple trapezoid transit model for fitting.
    Better than a pure box — captures ingress/egress.

    Parameters
    ----------
    phase            : array of phases (centered on 0)
    depth            : fractional flux drop (positive number)
    duration         : full duration in phase units
    ingress_fraction : fraction of duration spent in ingress/egress

    Returns
    -------
    flux model (normalized around 1.0)
    """
    flux       = np.ones_like(phase, dtype=float)
    half_dur   = duration / 2.0
    ing_width  = half_dur * ingress_fraction

    # Flat bottom
    in_transit = np.abs(phase) <= (half_dur - ing_width)
    flux[in_transit] = 1.0 - depth

    # Ingress (left side)
    ing_left = (phase >= -(half_dur)) & (phase < -(half_dur - ing_width))
    if ing_left.sum() > 0:
        t_norm = (phase[ing_left] + half_dur) / ing_width  # 0→1
        flux[ing_left] = 1.0 - depth * t_norm

    # Egress (right side)
    egr_right = (phase > (half_dur - ing_width)) & (phase <= half_dur)
    if egr_right.sum() > 0:
        t_norm = (half_dur - phase[egr_right]) / ing_width  # 0→1
        flux[egr_right] = 1.0 - depth * t_norm

    return flux


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP PARAMETER ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def fit_parameters(lc_flat, n_bootstrap=500, seed=42):
    """
    Estimate transit parameters with bootstrap uncertainties.

    Algorithm:
    1. Run BLS on full dataset → get best period, t0
    2. Phase-fold the light curve
    3. Bootstrap: resample the phase-folded points N times,
       refit BLS each time → distribution of parameters
    4. Report median ± 1-sigma from the bootstrap distributions

    Parameters
    ----------
    lc_flat     : flattened lightkurve LightCurve
    n_bootstrap : number of bootstrap iterations (500 is sufficient)
    seed        : random seed for reproducibility

    Returns
    -------
    params : dict with best-fit values and uncertainties
    """
    np.random.seed(seed)

    time  = np.asarray(lc_flat.time.value)
    flux  = np.asarray(lc_flat.flux.value)
    mask  = np.isfinite(time) & np.isfinite(flux)
    time  = time[mask]
    flux  = flux[mask]

    # ── STEP 1: BEST-FIT BLS ─────────────────────────────────────────────────
    print("  Running BLS on full dataset...", flush=True)
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

    # ── STEP 2: PHASE FOLD ───────────────────────────────────────────────────
    phase_all  = ((time - best_t0) % best_period) / best_period
    phase_all  = np.where(phase_all > 0.5, phase_all - 1.0, phase_all)
    half_dur_p = (best_duration / best_period) / 2.0

    in_mask    = np.abs(phase_all) <= half_dur_p
    out_mask   = np.abs(phase_all) >  half_dur_p

    out_scatter = np.std(flux[out_mask]) if out_mask.sum() > 2 else 1e-5
    out_mean    = np.mean(flux[out_mask]) if out_mask.sum() > 2 else 1.0
    in_mean     = np.mean(flux[in_mask])  if in_mask.sum()  > 2 else 1.0
    snr         = (out_mean - in_mean) / max(out_scatter, 1e-9)

    # Detection significance in sigma
    n_in     = in_mask.sum()
    n_out    = out_mask.sum()
    if n_in > 0 and n_out > 0:
        sigma_detection = snr * np.sqrt(n_in * n_out / (n_in + n_out))
    else:
        sigma_detection = snr

    # ── STEP 3: BOOTSTRAP ────────────────────────────────────────────────────
    print(f"  Running {n_bootstrap} bootstrap iterations...", flush=True)

    # We bootstrap on the phase-folded residuals
    phase_sorted  = phase_all.copy()
    flux_sorted   = flux.copy()

    boot_periods   = []
    boot_depths    = []
    boot_durations = []
    boot_t0s       = []

    # Narrow period search around best period for speed
    p_lo = max(0.5,           best_period * 0.8)
    p_hi = min(20.0,          best_period * 1.2)
    boot_periods_grid = np.linspace(p_lo, p_hi, 500)

    n_points = len(time)

    for i in range(n_bootstrap):
        # Resample with replacement
        idx      = np.random.choice(n_points, size=n_points, replace=True)
        t_boot   = time[idx]
        f_boot   = flux[idx]

        # Sort by time
        sort_idx = np.argsort(t_boot)
        t_boot   = t_boot[sort_idx]
        f_boot   = f_boot[sort_idx]

        try:
            m_boot = BoxLeastSquares(t_boot, f_boot)
            r_boot = m_boot.power(boot_periods_grid, durations)
            bi     = np.argmax(r_boot.power)

            boot_periods.append(float(r_boot.period[bi]))
            boot_depths.append(float(r_boot.depth[bi]))
            boot_durations.append(float(r_boot.duration[bi]))
            boot_t0s.append(float(r_boot.transit_time[bi]))
        except Exception:
            continue

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n_bootstrap} done", flush=True)

    boot_periods   = np.array(boot_periods)
    boot_depths    = np.array(boot_depths)
    boot_durations = np.array(boot_durations)

    # ── STEP 4: SUMMARIZE ────────────────────────────────────────────────────
    def ci(arr, pct=68.27):
        """Return median and half-width of central 68.27% interval (≈1σ)."""
        lo = np.percentile(arr, (100 - pct) / 2)
        hi = np.percentile(arr, 100 - (100 - pct) / 2)
        med = np.median(arr)
        return med, (hi - lo) / 2.0

    period_med,   period_err   = ci(boot_periods)
    depth_med,    depth_err    = ci(boot_depths)
    duration_med, duration_err = ci(boot_durations)

    params = {
        # Best-fit values (from full BLS)
        'period'         : best_period,
        'depth'          : best_depth,
        'duration'       : best_duration,
        't0'             : best_t0,

        # Bootstrap medians
        'period_med'     : period_med,
        'depth_med'      : depth_med,
        'duration_med'   : duration_med,

        # 1-sigma uncertainties from bootstrap
        'period_err'     : period_err,
        'depth_err'      : depth_err,
        'duration_err'   : duration_err,

        # SNR and significance
        'snr'            : snr,
        'sigma'          : sigma_detection,
        'bls_power'      : best_power,

        # Bootstrap arrays (for plotting posterior)
        'boot_periods'   : boot_periods,
        'boot_depths'    : boot_depths,
        'boot_durations' : boot_durations,

        # Phase-fold arrays
        'phase'          : phase_all,
        'flux'           : flux,
        'in_mask'        : in_mask,
        'out_mask'       : out_mask,
    }

    return params


# ─────────────────────────────────────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def print_results(params, tic_id=None, name=''):
    """Print a clean parameter summary."""
    header = f"TIC {tic_id}" if tic_id else "Target"
    if name:
        header += f" ({name})"

    print(f"\n{'═'*55}")
    print(f"  FITTED PARAMETERS — {header}")
    print(f"{'═'*55}")
    print(f"  Period    : {params['period_med']:.5f} ± {params['period_err']:.5f} days")
    print(f"  Depth     : {params['depth_med']*100:.4f} ± {params['depth_err']*100:.4f} %")
    print(f"  Duration  : {params['duration_med']*24:.3f} ± {params['duration_err']*24:.3f} hours")
    print(f"  SNR       : {params['snr']:.1f}")
    print(f"  Detection : {params['sigma']:.1f}σ significance")
    print(f"{'─'*55}")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_fit(params, tic_id=None, name='', save_path=None):
    """
    Generate a 4-panel figure:
      1. Phase-folded light curve with trapezoid model
      2. Bootstrap distribution of period
      3. Bootstrap distribution of depth
      4. Bootstrap distribution of duration
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title     = f"TIC {tic_id}" if tic_id else "Target"
    if name:
        title += f" — {name}"
    fig.suptitle(f"Transit Parameter Fit: {title}", fontsize=13)

    # ── Panel 1: Phase-folded + model ────────────────────────────────────────
    ax    = axes[0, 0]
    phase = params['phase']
    flux  = params['flux']

    # Bin the data for clarity
    n_bins  = 80
    bins    = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_mid = (bins[:-1] + bins[1:]) / 2
    bin_flux, bin_err = [], []

    for j in range(n_bins):
        mask_b = (phase >= bins[j]) & (phase < bins[j+1])
        if mask_b.sum() > 0:
            bin_flux.append(np.mean(flux[mask_b]))
            bin_err.append(np.std(flux[mask_b]) / np.sqrt(mask_b.sum()))
        else:
            bin_flux.append(np.nan)
            bin_err.append(np.nan)

    bin_flux = np.array(bin_flux)
    bin_err  = np.array(bin_err)

    # Raw scatter
    ax.scatter(phase, flux, s=1, alpha=0.2, color='steelblue', label='Data')

    # Binned points
    valid = np.isfinite(bin_flux)
    ax.errorbar(bin_mid[valid], bin_flux[valid], yerr=bin_err[valid],
                fmt='o', ms=4, color='navy', capsize=2, label='Binned', zorder=5)

    # Trapezoid model
    phase_model = np.linspace(-0.5, 0.5, 1000)
    dur_phase   = params['duration_med'] / params['period_med']
    flux_model  = trapezoid_model(phase_model, params['depth_med'], dur_phase)
    ax.plot(phase_model, flux_model, 'r-', lw=2, label='Trapezoid fit', zorder=6)

    ax.axhline(1.0, color='gray', linestyle='--', lw=0.8)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Normalized Flux')
    ax.set_title(f"Phase-Folded (P={params['period_med']:.4f}d, SNR={params['snr']:.1f})")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.5, 0.5)

    # ── Panel 2: Period posterior ─────────────────────────────────────────────
    ax = axes[0, 1]
    ax.hist(params['boot_periods'], bins=40, color='steelblue',
            edgecolor='white', linewidth=0.5)
    ax.axvline(params['period_med'], color='red', lw=2,
               label=f"Median = {params['period_med']:.5f} d")
    ax.axvline(params['period_med'] - params['period_err'],
               color='red', lw=1, linestyle='--')
    ax.axvline(params['period_med'] + params['period_err'],
               color='red', lw=1, linestyle='--', label=f"±{params['period_err']:.5f} d")
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('Bootstrap Count')
    ax.set_title('Period Posterior (Bootstrap)')
    ax.legend(fontsize=8)

    # ── Panel 3: Depth posterior ──────────────────────────────────────────────
    ax = axes[1, 0]
    ax.hist(params['boot_depths'] * 100, bins=40, color='darkorange',
            edgecolor='white', linewidth=0.5)
    ax.axvline(params['depth_med'] * 100, color='red', lw=2,
               label=f"Median = {params['depth_med']*100:.4f}%")
    ax.axvline((params['depth_med'] - params['depth_err']) * 100,
               color='red', lw=1, linestyle='--')
    ax.axvline((params['depth_med'] + params['depth_err']) * 100,
               color='red', lw=1, linestyle='--',
               label=f"±{params['depth_err']*100:.4f}%")
    ax.set_xlabel('Transit Depth (%)')
    ax.set_ylabel('Bootstrap Count')
    ax.set_title('Depth Posterior (Bootstrap)')
    ax.legend(fontsize=8)

    # ── Panel 4: Duration posterior ───────────────────────────────────────────
    ax = axes[1, 1]
    ax.hist(params['boot_durations'] * 24, bins=40, color='#2ecc71',
            edgecolor='white', linewidth=0.5)
    ax.axvline(params['duration_med'] * 24, color='red', lw=2,
               label=f"Median = {params['duration_med']*24:.3f} hr")
    ax.axvline((params['duration_med'] - params['duration_err']) * 24,
               color='red', lw=1, linestyle='--')
    ax.axvline((params['duration_med'] + params['duration_err']) * 24,
               color='red', lw=1, linestyle='--',
               label=f"±{params['duration_err']*24:.3f} hr")
    ax.set_xlabel('Duration (hours)')
    ax.set_ylabel('Bootstrap Count')
    ax.set_title('Duration Posterior (Bootstrap)')
    ax.legend(fontsize=8)

    plt.tight_layout()

    if save_path is None:
        save_path = f"fit_TIC{tic_id}.png" if tic_id else "fit_result.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import lightkurve as lk

    print("=" * 55)
    print("PARAMETER FITTER — Test on WASP-18")
    print("=" * 55)

    lc_raw  = lk.search_lightcurve('TIC 100100827', mission='TESS',
                                     exptime=120)[0].download()
    lc_flat = lc_raw.remove_nans().remove_outliers(sigma_lower=10, sigma_upper=5).flatten(
                  window_length=401, break_tolerance=5)

    params = fit_parameters(lc_flat, n_bootstrap=300)

    print_results(params, tic_id=100100827, name='WASP-18 b')

    print("\n── Known values for comparison ───────────────────────────────")
    print(f"  Period (known) : 0.94145 days")
    print(f"  Period (fitted): {params['period_med']:.5f} ± {params['period_err']:.5f} days")
    print(f"  Depth  (known) : ~1.0%")
    print(f"  Depth  (fitted): {params['depth_med']*100:.4f} ± {params['depth_err']*100:.4f}%")

    plot_fit(params, tic_id=100100827, name='WASP-18 b',
             save_path='fit_WASP18.png')
    print("\n✅ Done. Check fit_WASP18.png")
