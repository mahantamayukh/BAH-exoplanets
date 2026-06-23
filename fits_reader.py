#!/usr/bin/env python3
"""
TESS FITS File Reader — Full Information Dumper
================================================

What this does:
  - Opens any TESS .fits light-curve file
  - Prints EVERY piece of information in the file:
      * All extensions (HDUs) and their shapes
      * All header keywords (not just the curated subset) for every HDU
      * All columns in the LIGHTCURVE table with full descriptions
      * All pixel values in the APERTURE image, decoded
      * Summary statistics for every numeric column
      * First 5 rows of the LIGHTCURVE table
  - Optionally saves a PNG plot of the light curve

Usage:
    python3 fits_reader.py                              # pops up a file picker
    python3 fits_reader.py tess2018...s_lc.fits         # pass filename as arg

Dependencies:
    pip install astropy numpy matplotlib
    (tkinter comes built-in with Python on macOS — no install needed)
"""

import os
import sys
import numpy as np
from astropy.io import fits

# Import the interactive plotting module (must be in the same directory
# or on the Python path)
try:
    from interactive_plot import plot_lightcurve_interactive
    _HAS_INTERACTIVE_PLOT = True
except ImportError:
    _HAS_INTERACTIVE_PLOT = False


# ---------------------------------------------------------------------------
# File picker popup (tkinter)
# ---------------------------------------------------------------------------

def pick_fits_file():
    """Open a native OS file picker dialog and return the chosen path.

    Falls back to typing the path if tkinter is unavailable or the user
    cancels the dialog.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("  (tkinter not available — falling back to typed input)")
        return input(
            "Enter path to the .fits file\n"
            "  (e.g. tess2018...s_lc.fits): "
        ).strip().strip('"').strip("'")

    # Hide the root window so only the dialog is visible
    root = tk.Tk()
    root.withdraw()
    # Bring the dialog to the front on macOS
    try:
        root.attributes('-topmost', True)
    except Exception:
        pass

    path = filedialog.askopenfilename(
        title="Select a TESS .fits light-curve file",
        filetypes=[
            ("TESS FITS files", "*.fits"),
            ("FITS files", "*.fits *.fit"),
            ("All files", "*.*"),
        ],
        initialdir=os.getcwd(),
    )

    root.destroy()

    if not path:
        print("\nNo file selected in the dialog.")
        fallback = input(
            "Type a path manually, or press Enter to quit: "
        ).strip().strip('"').strip("'")
        return fallback

    return path


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def banner(title, char='='):
    line = char * 78
    print("\n" + line)
    print(f" {title}")
    print(line)


def subbanner(title, char='-'):
    line = char * 78
    print("\n" + line)
    print(f" {title}")
    print(line)


# ---------------------------------------------------------------------------
# Dump the full header (every keyword)
# ---------------------------------------------------------------------------

def dump_full_header(hdu, hdu_index):
    """Print every keyword in a header, including comments and HISTORY."""
    subbanner(f"HDU[{hdu_index}] FULL HEADER  (name={hdu.name!r})")
    header = hdu.header
    print(f"Total cards: {len(header)}\n")
    print(f"{'#':>3}  {'Keyword':<12} {'Value':<40} {'Comment'}")
    print("-" * 100)
    for i, card in enumerate(header.cards):
        keyword = card.keyword
        value = str(card.value)
        comment = card.comment or ''
        # Truncate very long values
        if len(value) > 38:
            value = value[:35] + '...'
        print(f"{i+1:>3}  {keyword:<12} {value:<40} {comment}")


# ---------------------------------------------------------------------------
# Dump the LIGHTCURVE binary table
# ---------------------------------------------------------------------------

def dump_lightcurve_table(hdu):
    """Show all columns, full stats, and first/last few rows."""
    data = hdu.data
    if data is None:
        print("  (No data table in this HDU.)")
        return

    subbanner("LIGHTCURVE TABLE — ALL COLUMNS")
    print(f"Total rows     : {len(data):,}")
    print(f"Total columns  : {len(data.columns)}\n")

    print(f"{'#':>3}  {'Column':<22} {'Format':<12} {'Unit':<10} {'Description'}")
    print("-" * 100)

    # TESS column descriptions (best-effort)
    column_docs = {
        'TIME':            'Time in BJD-2457000 (days)',
        'TIMECORR':        'Time correction applied (days)',
        'CADENCENO':       'Unique cadence number',
        'SAP_FLUX':        'Raw brightness from aperture photometry (e-/s)',
        'SAP_FLUX_ERR':    'Uncertainty on SAP_FLUX',
        'SAP_BKG':         'Background flux inside aperture (e-/s)',
        'SAP_BKG_ERR':     'Uncertainty on SAP_BKG',
        'PDCSAP_FLUX':     'CLEANED brightness (PDC-processed, use for ML)',
        'PDCSAP_FLUX_ERR': 'Uncertainty on PDCSAP_FLUX',
        'SAP_QUALITY':     'Quality flags (0 = good, bitmask otherwise)',
        'PSF_CENTR1':      'PSF centroid column (pixels)',
        'PSF_CENTR1_ERR':  'Uncertainty on PSF_CENTR1',
        'PSF_CENTR2':      'PSF centroid row (pixels)',
        'PSF_CENTR2_ERR':  'Uncertainty on PSF_CENTR2',
        'MOM_CENTR1':      'Moment centroid column (pixels)',
        'MOM_CENTR1_ERR':  'Uncertainty on MOM_CENTR1',
        'MOM_CENTR2':      'Moment centroid row (pixels)',
        'MOM_CENTR2_ERR':  'Uncertainty on MOM_CENTR2',
        'POS_CORR1':       'Column drift correction (pixels)',
        'POS_CORR2':       'Row drift correction (pixels)',
    }

    for i, col in enumerate(data.columns):
        unit = str(col.unit) if col.unit else ''
        fmt = col.format
        desc = column_docs.get(col.name, '(pipeline column)')
        print(f"{i+1:>3}  {col.name:<22} {fmt:<12} {unit:<10} {desc}")

    # Full statistics on every numeric column
    subbanner("LIGHTCURVE — STATISTICS FOR EVERY NUMERIC COLUMN")
    print(f"{'Column':<22} {'Min':>14} {'Max':>14} {'Mean':>14} "
          f"{'Median':>14} {'Finite':>10} {'Total':>8}")
    print("-" * 100)
    for col in data.columns:
        try:
            arr = np.array(data[col.name], dtype=float)
        except (ValueError, TypeError):
            print(f"{col.name:<22}  (non-numeric, skipped)")
            continue
        total = len(arr)
        finite_mask = np.isfinite(arr)
        n_finite = int(finite_mask.sum())
        if n_finite == 0:
            print(f"{col.name:<22}  all NaN/inf")
            continue
        finite = arr[finite_mask]
        print(f"{col.name:<22} {finite.min():>14.4g} {finite.max():>14.4g} "
              f"{finite.mean():>14.4g} {np.median(finite):>14.4g} "
              f"{n_finite:>10,} {total:>8,}")

    # First 5 rows
    subbanner("LIGHTCURVE — FIRST 5 ROWS (all columns)")
    print("\n".join(
        "  " + " | ".join(f"{col.name}={data[col.name][i]}" for col in data.columns)
        for i in range(min(5, len(data)))
    ))

    # Last 5 rows
    if len(data) > 5:
        subbanner("LIGHTCURVE — LAST 5 ROWS (all columns)")
        print("\n".join(
            "  " + " | ".join(f"{col.name}={data[col.name][i]}" for col in data.columns)
            for i in range(max(0, len(data) - 5), len(data))
        ))


# ---------------------------------------------------------------------------
# Dump the APERTURE image (every raw value + decoded meaning)
# ---------------------------------------------------------------------------

def dump_aperture(hdu):
    """Show the raw pixel values AND the decoded bit-flag meaning."""
    data = hdu.data
    if data is None:
        print("  (No image data in this HDU.)")
        return

    subbanner("APERTURE IMAGE — RAW PIXEL VALUES")
    print(f"Shape: {data.shape}  (rows x columns)\n")
    print("Raw integer values (one per pixel):")
    for r, row in enumerate(data):
        print(f"  row {r:>2}:  " + "  ".join(f"{int(v):>3}" for v in row))

    subbanner("APERTURE IMAGE — DECODED MEANING OF EACH UNIQUE VALUE")
    unique_vals = sorted(set(np.asarray(data).ravel().tolist()))
    print(f"Unique values found: {unique_vals}\n")
    print("Bit-flag legend:")
    print("  bit 0  (value   1) -> pixel is in the aperture (collected flux)")
    print("  bit 1  (value   2) -> pixel used for flux-weighted centroid")
    print("  bit 2  (value   4) -> pixel used for moment centroid")
    print("  bit 5  (value  32) -> pixel was on the CCD frame\n")
    print(f"{'Value':<8} {'Binary':<10} {'Decoded meaning'}")
    print("-" * 70)
    for v in unique_vals:
        bits = []
        if v & 1:   bits.append("in_aperture")
        if v & 2:   bits.append("flux_centroid")
        if v & 4:   bits.append("moment_centroid")
        if v & 8:   bits.append("bit3_unknown")
        if v & 16:  bits.append("bit4_unknown")
        if v & 32:  bits.append("on_ccd")
        if v & 64:  bits.append("bit6_unknown")
        decoded = " + ".join(bits) if bits else "0 (no flags)"
        print(f"{v:<8} {bin(v):<10} {decoded}")

    subbanner("APERTURE IMAGE — ASCII VISUAL")
    print("Legend:  '#' = pixel in aperture (bit 0 set)")
    print("         '.' = pixel not in aperture")
    width = data.shape[1]
    print("  " + "-" * (width + 2))
    for row in data:
        line = "  |"
        for val in row:
            line += "#" if (int(val) & 1) else "."
        line += "|"
        print(line)
    print("  " + "-" * (width + 2))


# ---------------------------------------------------------------------------
# Optional: save a light-curve plot
# ---------------------------------------------------------------------------

def plot_lightcurve(local_path, tic_id, sector, denoise_method=None):
    """Generate an interactive Plotly chart (opens in browser) or fallback PNG."""
    if _HAS_INTERACTIVE_PLOT:
        with fits.open(local_path) as hdul:
            if len(hdul) < 2 or hdul[1].data is None:
                print("  No light curve data to plot.")
                return None
            data = hdul[1].data
            t = np.array(data['TIME'], dtype=float)
            f = np.array(data['PDCSAP_FLUX'], dtype=float)
            sap = np.array(data['SAP_FLUX'], dtype=float) if 'SAP_FLUX' in data.columns else None
            qual = np.array(data['QUALITY']) if 'QUALITY' in data.columns else None

        out_html = f"tic_{tic_id}_sector_{sector}_interactive.html"
        return plot_lightcurve_interactive(
            time=t, pdcsap_flux=f, sap_flux=sap, quality=qual,
            tic_id=str(tic_id), sector=str(sector),
            output_html=out_html, auto_open=True,
            denoise_method=denoise_method,
        )
    else:
        return _plot_lightcurve_png(local_path, tic_id, sector)


def _plot_lightcurve_png(local_path, tic_id, sector):
    """Fallback: save a static PNG plot using matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping plot)")
        return None

    with fits.open(local_path) as hdul:
        data = hdul[1].data
        t = np.array(data['TIME'], dtype=float)
        f = np.array(data['PDCSAP_FLUX'], dtype=float)

    mask = np.isfinite(t) & np.isfinite(f)
    if mask.sum() == 0:
        print("  No finite data points to plot.")
        return None
    t = t[mask]; f = f[mask]
    median = np.median(f) or 1.0
    f_norm = (f / median) * 100

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.plot(t, f_norm, '.', markersize=1, alpha=0.5)
    ax.set_xlabel('Time (BJD - 2457000, days)')
    ax.set_ylabel('PDCSAP_FLUX (normalized, %)')
    ax.set_title(f"TIC {tic_id} - Sector {sector}")
    ax.grid(True, alpha=0.3)

    out_png = f"tic_{tic_id}_sector_{sector}.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    banner("TESS FITS FILE READER — Full Information Dumper")

    # --- Get path ---
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"\nUsing file from command line: {path}")
    else:
        print("\nOpening file picker dialog...")
        print("(Choose any .fits file on your Mac and click Open.)")
        path = pick_fits_file()

    if not path:
        print("\nNo file selected. Exiting.")
        sys.exit(0)

    if not os.path.exists(path):
        print(f"\nERROR: file not found: {path}")
        print("Tip: pass the filename as an argument, e.g.:")
        print("  python3 fits_reader.py tess2018...s_lc.fits")
        sys.exit(1)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\nFile: {path}")
    print(f"Size: {size_mb:.2f} MB")

    # --- Open and dump ---
    with fits.open(path) as hdul:
        banner(f"OPENED — {len(hdul)} HDUs (extensions)")
        for i, hdu in enumerate(hdul):
            shape = getattr(hdu.data, 'shape', 'no data')
            print(f"  HDU[{i}]  name={hdu.name!r:<15}  data shape={shape}")

        # Dump every header of every HDU
        for i, hdu in enumerate(hdul):
            dump_full_header(hdu, i)

        # Dump LIGHTCURVE table if present
        for hdu in hdul:
            if hdu.name == 'LIGHTCURVE' and hdu.data is not None:
                dump_lightcurve_table(hdu)
                break

        # Dump APERTURE image if present
        for hdu in hdul:
            if hdu.name == 'APERTURE' and hdu.data is not None:
                dump_aperture(hdu)
                break

    # --- Optional plot ---
    plot_type = "interactive HTML" if _HAS_INTERACTIVE_PLOT else "PNG"
    answer = input(f"\nGenerate {plot_type} plot of the light curve? [y/N]: ").strip().lower()
    if answer == 'y':
        # Extract TIC ID and sector from filename if possible
        import re
        tic_match = re.search(r'-(\d{16})-', os.path.basename(path))
        tic_id = tic_match.group(1) if tic_match else 'unknown'
        sector_match = re.search(r'-s(\d+)-', os.path.basename(path))
        sector = str(int(sector_match.group(1))) if sector_match else '??'

        # Ask which denoising method to apply
        denoise_method = None
        if _HAS_INTERACTIVE_PLOT:
            print("\n  Denoising options:")
            print("    0) None - plot raw PDCSAP_FLUX")
            print("    1) Gaussian Process regression (smooths noise, preserves transits)")
            choice = input("  Choose [0/1, default=0]: ").strip()
            if choice == '1':
                denoise_method = 'gp'

        out = plot_lightcurve(path, tic_id, sector, denoise_method=denoise_method)
        if out:
            print(f"Plot saved to: {os.path.abspath(out)}")
            if _HAS_INTERACTIVE_PLOT:
                print("(Interactive chart should open in your browser automatically)")

    print("\nDone.")


if __name__ == "__main__":
    main()
