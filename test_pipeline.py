"""
Pipeline Foundation Test
Tests: download → clean → BLS detect on WASP-18 (TIC 100100827)
Expected: period ~0.94 days, depth ~1%
"""

import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
import warnings
warnings.filterwarnings('ignore')

# ── 1. DOWNLOAD ───────────────────────────────────────────────
print("=" * 55)
print("STEP 1: Downloading WASP-18 light curve from MAST...")
print("=" * 55)

search = lk.search_lightcurve('TIC 100100827', mission='TESS', exptime=120)
print(f"Found {len(search)} observations")
print(search)

# Download the first available sector
lc_raw = search[0].download()
print(f"\nDownloaded: {len(lc_raw)} data points")
print(f"Time range: {lc_raw.time.value[0]:.2f} to {lc_raw.time.value[-1]:.2f} days")

# ── 2. CLEAN ──────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Cleaning light curve...")
print("=" * 55)

lc_clean = lc_raw.remove_nans()
lc_clean = lc_clean.remove_outliers(sigma=5)
lc_flat  = lc_clean.flatten(window_length=401, break_tolerance=5)

print(f"Points after cleaning: {len(lc_flat)}")

# ── 3. BLS PERIOD SEARCH ──────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Running Box Least Squares...")
print("=" * 55)

time  = np.asarray(lc_flat.time.value)
flux  = np.asarray(lc_flat.flux.value)
mask  = np.isfinite(time) & np.isfinite(flux)
time  = time[mask]
flux  = flux[mask]

periods   = np.linspace(0.5, 20.0, 10000)
durations = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20])

model   = BoxLeastSquares(time, flux)
results = model.power(periods, durations)

best_idx      = np.argmax(results.power)
best_period   = float(results.period[best_idx])
best_duration = float(results.duration[best_idx])
best_depth    = float(results.depth[best_idx])
best_t0       = float(results.transit_time[best_idx])
best_power    = float(results.power[best_idx])

print(f"\n{'─'*40}")
print(f"  Best Period   : {best_period:.4f} days")
print(f"  Expected      : ~0.94 days")
print(f"  Best Duration : {best_duration:.4f} days")
print(f"  Best Depth    : {best_depth:.6f}")
print(f"  Expected depth: ~0.01 (1%)")
print(f"  BLS Power     : {best_power:.4f}")
print(f"{'─'*40}")

# ── 4. SNR (quick estimate) ───────────────────────────────────
# SNR = depth / (scatter of out-of-transit flux)
folded      = lc_flat.fold(period=best_period, epoch_time=best_t0)
phase       = folded.phase.value
flux_folded = folded.flux.value

half_dur    = best_duration / (2 * best_period)
out_mask    = np.abs(phase) > half_dur
in_mask     = np.abs(phase) <= half_dur

out_scatter = np.std(flux_folded[out_mask])
in_mean     = np.mean(flux_folded[in_mask])
out_mean    = np.mean(flux_folded[out_mask])
snr         = (out_mean - in_mean) / out_scatter

print(f"\n  Estimated SNR : {snr:.1f}")
print(f"  (>7 is generally considered a detection)")

# ── 5. PLOTS ──────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4: Generating plots...")
print("=" * 55)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle("WASP-18 (TIC 100100827) — Pipeline Test", fontsize=14)

# Raw light curve
axes[0].scatter(lc_raw.time.value, lc_raw.flux.value, s=1, alpha=0.4, color='steelblue')
axes[0].set_title("Raw Light Curve")
axes[0].set_ylabel("Flux (e-/s)")
axes[0].set_xlabel("Time (BTJD days)")

# BLS periodogram
axes[1].plot(results.period, results.power, linewidth=0.8, color='darkorange')
axes[1].axvline(best_period, color='red', linestyle='--', linewidth=1.5,
                label=f"Best period = {best_period:.4f} d")
axes[1].set_title("BLS Periodogram")
axes[1].set_xlabel("Period (days)")
axes[1].set_ylabel("BLS Power")
axes[1].legend()

# Phase-folded light curve
axes[2].scatter(phase, flux_folded, s=2, alpha=0.4, color='steelblue')
axes[2].axhline(1.0, linestyle='--', color='gray', linewidth=1)
axes[2].set_title(f"Phase-Folded Light Curve (P = {best_period:.4f} days, SNR = {snr:.1f})")
axes[2].set_xlabel("Phase")
axes[2].set_ylabel("Normalized Flux")

plt.tight_layout()
plt.savefig("wasp18_pipeline_test.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Plot saved as wasp18_pipeline_test.png")

# ── 6. PASS / FAIL CHECK ─────────────────────────────────────
print("\n" + "=" * 55)
print("RESULT")
print("=" * 55)

period_ok = 0.85 < best_period < 1.05   # within ~10% of 0.94
depth_ok  = best_depth > 0.005           # at least 0.5% depth
snr_ok    = snr > 7

if period_ok and depth_ok and snr_ok:
    print("✅ PASS — pipeline working correctly")
    print("   Period, depth, and SNR all within expected range")
    print("   Ready to build the classifier on top of this")
else:
    print("⚠️  CHECK RESULTS:")
    print(f"   Period check (0.85–1.05 d): {'✅' if period_ok else '❌'} got {best_period:.4f}")
    print(f"   Depth check  (> 0.005)    : {'✅' if depth_ok  else '❌'} got {best_depth:.6f}")
    print(f"   SNR check    (> 7)        : {'✅' if snr_ok    else '❌'} got {snr:.1f}")