"""
TESS FITS Light Curve Cleaner (BLS Ready)
Author: Mayukh

Features
--------
- File picker for FITS file
- Reads TESS SPOC light curves
- Removes NaNs
- Removes outliers
- Flattens long-term trends
- Saves cleaned CSV beside original FITS
- Saves cleaned FITS beside original FITS
- Plots raw and cleaned curves
"""

import os
import tkinter as tk
from tkinter import filedialog

import lightkurve as lk
import matplotlib.pyplot as plt
import pandas as pd


def select_fits_file():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select TESS FITS File",
        filetypes=[
            ("FITS Files", "*.fits"),
            ("All Files", "*.*")
        ]
    )

    return file_path


def load_lightcurve(fits_file):

    print("\nLoading FITS file...")

    try:
        lc = lk.read(fits_file)

        print(f"Total Data Points: {len(lc)}")

        return lc

    except Exception as e:
        print(f"\nError loading FITS file:\n{e}")
        raise


def clean_lightcurve(lc):

    print("\nCleaning Light Curve...")

    # Remove NaNs
    print("Removing NaNs...")
    lc_clean = lc.remove_nans()

    # Remove extreme outliers
    print("Removing Outliers...")
    lc_clean = lc_clean.remove_outliers(sigma=5)

    # Remove long-term trends while preserving transits
    print("Flattening Light Curve...")

    lc_flat = lc_clean.flatten(
        window_length=401,
        break_tolerance=5
    )

    print(f"Remaining Points: {len(lc_flat)}")

    return lc_flat


def save_csv(cleaned_lc, fits_file):

    data = {
        "time": cleaned_lc.time.value,
        "flux": cleaned_lc.flux.value
    }

    try:
        if cleaned_lc.flux_err is not None:
            data["flux_err"] = cleaned_lc.flux_err.value
    except Exception:
        pass

    df = pd.DataFrame(data)

    fits_dir = os.path.dirname(fits_file)

    base_name = os.path.splitext(
        os.path.basename(fits_file)
    )[0]

    output_csv = os.path.join(
        fits_dir,
        f"{base_name}_cleaned.csv"
    )

    df.to_csv(output_csv, index=False)

    print("\nSaved cleaned CSV:")
    print(output_csv)


def save_cleaned_fits(cleaned_lc, fits_file):

    fits_dir = os.path.dirname(fits_file)

    base_name = os.path.splitext(
        os.path.basename(fits_file)
    )[0]

    output_fits = os.path.join(
        fits_dir,
        f"{base_name}_cleaned.fits"
    )

    try:
        cleaned_lc.to_fits(
            path=output_fits,
            overwrite=True
        )

        print("\nSaved cleaned FITS:")
        print(output_fits)

    except Exception as e:
        print("\nCould not save cleaned FITS:")
        print(e)


def plot_lightcurves(raw_lc, cleaned_lc):

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True
    )

    # Raw light curve
    axes[0].scatter(
        raw_lc.time.value,
        raw_lc.flux.value,
        s=1,
        alpha=0.6
    )

    axes[0].set_title("Raw Light Curve")
    axes[0].set_ylabel("Flux")

    # Cleaned light curve
    axes[1].scatter(
        cleaned_lc.time.value,
        cleaned_lc.flux.value,
        s=1,
        alpha=0.6
    )

    axes[1].axhline(
        y=1.0,
        linestyle="--"
    )

    axes[1].set_title("Cleaned & Flattened Light Curve")
    axes[1].set_xlabel("Time (Days)")
    axes[1].set_ylabel("Normalized Flux")

    plt.tight_layout()
    plt.show()


def main():

    print("=" * 60)
    print("TESS Light Curve Cleaner (BLS Ready)")
    print("=" * 60)

    fits_file = select_fits_file()

    if not fits_file:
        print("\nNo file selected.")
        return

    print("\nSelected File:")
    print(fits_file)

    raw_lc = load_lightcurve(fits_file)

    cleaned_lc = clean_lightcurve(raw_lc)

    save_csv(cleaned_lc, fits_file)

    save_cleaned_fits(cleaned_lc, fits_file)

    plot_lightcurves(raw_lc, cleaned_lc)

    print("\nFinished Successfully.")


if __name__ == "__main__":
    main()
