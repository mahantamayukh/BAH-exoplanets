"""
TESS BLS Transit Detector
BLS-ready version

Input:
    *_cleaned.fits

Output:
    - Best transit candidate
    - BLS periodogram
    - Folded light curve

Author: Mayukh
"""

import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

from astropy.timeseries import BoxLeastSquares


# ==========================================================
# FILE PICKER
# ==========================================================

def choose_file():

    root = tk.Tk()
    root.withdraw()

    filename = filedialog.askopenfilename(
        title="Select Cleaned FITS File",
        filetypes=[
            ("FITS Files", "*.fits"),
            ("All Files", "*.*")
        ]
    )

    return filename


# ==========================================================
# LOAD LIGHT CURVE
# ==========================================================

def load_lightcurve(fits_file):

    print("\nLoading cleaned light curve...")

    lc = lk.read(fits_file)

    lc = lc.remove_nans()

    return lc


# ==========================================================
# BLS SEARCH
# ==========================================================

def run_bls(lc):

    print("\nRunning Box Least Squares...")

    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)

    mask = np.isfinite(time) & np.isfinite(flux)

    time = time[mask]
    flux = flux[mask]

    # --------------------------------------------------
    # PERIOD SEARCH RANGE
    # --------------------------------------------------

    min_period = 0.5
    max_period = 20.0

    periods = np.linspace(
        min_period,
        max_period,
        10000
    )

    # --------------------------------------------------
    # SAFE TRANSIT DURATIONS
    # Must always be < min_period
    # --------------------------------------------------

    durations = np.array([
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
        0.15,
        0.20
    ])

    model = BoxLeastSquares(
        time,
        flux
    )

    results = model.power(
        periods,
        durations
    )

    best_index = np.argmax(results.power)

    best_period = float(results.period[best_index])
    best_duration = float(results.duration[best_index])
    best_depth = float(results.depth[best_index])
    best_t0 = float(results.transit_time[best_index])
    best_power = float(results.power[best_index])

    print("\n========== BEST CANDIDATE ==========")
    print(f"Period       : {best_period:.6f} days")
    print(f"Duration     : {best_duration:.6f} days")
    print(f"Depth        : {best_depth:.8f}")
    print(f"Transit Time : {best_t0:.6f}")
    print(f"BLS Power    : {best_power:.6f}")

    return (
        results,
        best_period,
        best_duration,
        best_depth,
        best_t0,
        best_power
    )


# ==========================================================
# PERIODIGRAM
# ==========================================================

def plot_periodogram(results):

    best_idx = np.argmax(results.power)

    plt.figure(figsize=(12, 5))

    plt.plot(
        results.period,
        results.power,
        linewidth=1
    )

    plt.axvline(
        results.period[best_idx],
        linestyle="--",
        linewidth=1,
        label=f"Best Period = {results.period[best_idx]:.5f} d"
    )

    plt.xlabel("Period (days)")
    plt.ylabel("BLS Power")
    plt.title("BLS Periodogram")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ==========================================================
# FOLDED LIGHT CURVE
# ==========================================================

def plot_folded_curve(
    lc,
    period,
    t0
):

    folded = lc.fold(
        period=period,
        epoch_time=t0
    )

    plt.figure(figsize=(12, 5))

    plt.scatter(
        folded.phase.value,
        folded.flux.value,
        s=2,
        alpha=0.5
    )

    plt.axhline(
        1.0,
        linestyle="--"
    )

    plt.xlabel("Phase")
    plt.ylabel("Normalized Flux")

    plt.title(
        f"Folded Light Curve\nPeriod = {period:.5f} days"
    )

    plt.tight_layout()
    plt.show()


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("TESS BLS Transit Detector")
    print("=" * 60)

    fits_file = choose_file()

    if not fits_file:
        print("\nNo file selected.")
        return

    print("\nSelected:")
    print(fits_file)

    lc = load_lightcurve(fits_file)

    (
        results,
        best_period,
        best_duration,
        best_depth,
        best_t0,
        best_power
    ) = run_bls(lc)

    plot_periodogram(results)

    plot_folded_curve(
        lc,
        best_period,
        best_t0
    )

    print("\nFinished Successfully.")


if __name__ == "__main__":
    main()