#!/usr/bin/env python3
"""
TESS Sector Curl File Explorer
==============================

What this does:
  1. Takes a TESS sector curl script (e.g. tesscurl_sector_1_lc.sh)
  2. Lists the stars (by TIC ID) available in that sector
  3. Asks the user which star they want to open
  4. Downloads the .fits file if it is not already present locally
  5. Opens the FITS file and explains all three extensions in plain English:
       - Extension 0 (Primary): header / observation metadata
       - Extension 1 (LIGHTCURVE): the actual brightness-vs-time table
       - Extension 2 (APERTURE): the pixel mask used to extract the flux
  6. Optionally plots the light curve as a PNG

Usage:
    python3 tess_explorer.py                  # interactive: pick mode (file picker, path, or sector)
    python3 tess_explorer.py tesscurl_sector_1_lc.sh    # use existing .sh file
    python3 tess_explorer.py 1                # shortcut: download sector 1

If no argument is given, you will be prompted for the mode:
  1) Browse for .sh file on disk via file picker popup
  2) Type the .sh file path manually
  3) Just give a sector number — auto-download from MAST
"""

import os
import re
import sys
import urllib.request
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
# File picker popup (tkinter) — for choosing the .sh file from device
# ---------------------------------------------------------------------------

def pick_sh_file():
    """Open a native OS file picker dialog and return the chosen .sh path.

    Falls back to typing the path if tkinter is unavailable or the user
    cancels the dialog.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("  (tkinter not available — falling back to typed input)")
        return input(
            "Enter the path to the TESS curl script\n"
            "  (e.g. tesscurl_sector_1_lc.sh): "
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
        title="Select a TESS curl script (.sh file)",
        filetypes=[
            ("TESS curl scripts", "*.sh"),
            ("Shell scripts", "*.sh *.bash"),
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
# Step 1 — Parse the curl script
# ---------------------------------------------------------------------------

def parse_curl_script(script_path):
    """Extract (filename, url, tic_id, sector) for every star in the script."""
    # Each line in the script looks like:
    #   curl -C - -L -o tess2018206045859-s0001-0000000278660115-0120-s_lc.fits https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/tess2018206045859-s0001-0000000278660115-0120-s_lc.fits
    pattern = re.compile(r'curl\s+-C\s+-\s+-L\s+-o\s+(\S+)\s+(https?://\S+)')

    entries = []
    with open(script_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = pattern.search(line)
            if not m:
                continue
            filename = m.group(1)
            url = m.group(2)

            # Decode the TESS filename convention:
            #   tess<timestamp>-s<SECTOR>-<TIC_ID padded to 16 digits>-<build>-s_lc.fits
            tic_match = re.search(r'-(\d{16})-', filename)
            tic_id = tic_match.group(1) if tic_match else 'unknown'
            sector_match = re.search(r'-s(\d+)-', filename)
            sector = str(int(sector_match.group(1))) if sector_match else '??'

            entries.append({
                'filename': filename,
                'url': url,
                'tic_id': tic_id,
                'sector': sector,
            })
    return entries


# ---------------------------------------------------------------------------
# Step 2 — Download a .fits file (only if not already present)
# ---------------------------------------------------------------------------

def download_file(url, local_path):
    """Download a single file with progress reporting."""
    print(f"  Downloading from: {url}")
    print(f"  Saving to:        {local_path}")
    urllib.request.urlretrieve(url, local_path)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"  Done. Size: {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# Step 3 — Explain each extension in plain English
# ---------------------------------------------------------------------------

def explain_primary_hdu(hdu):
    """Explain Extension 0 — the Primary HDU (header-only, no data table)."""
    print("\n" + "=" * 72)
    print("EXTENSION 0 — PRIMARY HDU  (header / observation metadata)")
    print("=" * 72)
    print("This is the 'cover page' of the FITS file. It contains NO data table,")
    print("only metadata describing the observation: which star, where, when,")
    print("and how it was processed.")
    print()

    header = hdu.header
    interesting_keys = [
        ('TELESCOP', 'Telescope used'),
        ('INSTRUME', 'Instrument / mission'),
        ('SECTOR',   'TESS sector number'),
        ('TICID',    'TIC ID of the target star'),
        ('CAMERA',   'TESS camera number (1-4)'),
        ('CCD',      'CCD number (1-4)'),
        ('RA_OBJ',   'Right Ascension of target (degrees)'),
        ('DEC_OBJ',  'Declination of target (degrees)'),
        ('TICVER',   'TIC catalog version'),
        ('TIMESYS',  'Time system'),
        ('BJDREFI',  'BJD reference date — integer part'),
        ('BJDREFF',  'BJD reference date — fractional part'),
        ('TIMEUNIT', 'Time unit'),
        ('DATE-OBS', 'Observation start date (UTC)'),
        ('DATE-END', 'Observation end date (UTC)'),
        ('CREATOR',  'Software that produced this file'),
        ('PROCVER',  'Processing pipeline version'),
    ]
    print(f"{'Keyword':<12} {'Value':<35} {'Description'}")
    print("-" * 80)
    for key, desc in interesting_keys:
        if key in header:
            val = header[key]
            print(f"{key:<12} {str(val):<35} {desc}")

    print(f"\nTotal header cards in this HDU: {len(header)}")


def explain_lightcurve_hdu(hdu):
    """Explain Extension 1 — the LIGHTCURVE binary table."""
    print("\n" + "=" * 72)
    print("EXTENSION 1 — LIGHTCURVE  (the actual brightness-vs-time table)")
    print("=" * 72)
    print("This is the real data. It is a binary table where each row is one")
    print("2-minute exposure. A full 27-day TESS sector gives ~18,000 rows")
    print("at 2-min cadence.")
    print()

    data = hdu.data
    if data is None:
        print("  (No data in this extension.)")
        return

    print(f"  Number of rows  (observations): {len(data):,}")
    print(f"  Number of columns:              {len(data.columns)}")
    print()

    print("Column guide (the most important ones are marked with *):")
    print("-" * 80)
    print(f"{'Column':<22} {'Unit':<10} {'Description'}")
    print("-" * 80)

    column_info = {
        'TIME':            ('d',    '* Time in BJD-2457000 (days)'),
        'TIMECORR':        ('d',    'Correction applied to time (days)'),
        'CADENCENO':       ('',     'Unique cadence number'),
        'SAP_FLUX':        ('e-/s', '* Raw brightness from aperture photometry'),
        'SAP_FLUX_ERR':    ('e-/s', 'Uncertainty on SAP_FLUX'),
        'SAP_BKG':         ('e-/s', 'Background flux inside aperture'),
        'SAP_BKG_ERR':     ('e-/s', 'Uncertainty on background'),
        'PDCSAP_FLUX':     ('e-/s', '* CLEANED brightness (use this one!)'),
        'PDCSAP_FLUX_ERR': ('e-/s', 'Uncertainty on PDCSAP_FLUX'),
        'SAP_QUALITY':     ('',     'Quality flags (0 = good, nonzero = bad)'),
        'PSF_CENTR1':      ('pix',  'PSF centroid — column position'),
        'PSF_CENTR2':      ('pix',  'PSF centroid — row position'),
        'MOM_CENTR1':      ('pix',  'Moment centroid — column position'),
        'MOM_CENTR2':      ('pix',  'Moment centroid — row position'),
        'POS_CORR1':       ('pix',  'Column position correction (drift)'),
        'POS_CORR2':       ('pix',  'Row position correction (drift)'),
    }

    for col in data.columns:
        desc = column_info.get(col.name)
        if desc:
            unit, description = desc
        else:
            unit = str(col.unit) if col.unit else ''
            description = '(additional pipeline column)'
        print(f"{col.name:<22} {unit:<10} {description}")

    print("\nKey statistics on the most important columns:")
    for col_name in ['TIME', 'SAP_FLUX', 'PDCSAP_FLUX']:
        if col_name not in data.columns:
            continue
        arr = np.array(data[col_name], dtype=float)
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            print(f"  {col_name:<15} all NaN")
            continue
        print(f"  {col_name:<15} min={finite.min():.4f}  max={finite.max():.4f}  "
              f"mean={finite.mean():.4f}  n_finite={len(finite):,}/{len(arr):,}")

    if 'SAP_QUALITY' in data.columns:
        q = np.array(data['SAP_QUALITY'])
        n_good = int(np.sum(q == 0))
        n_bad  = int(np.sum(q != 0))
        print(f"  SAP_QUALITY     {n_good:,} good cadences, {n_bad:,} flagged as bad")

    if 'TIME' in data.columns:
        t = np.array(data['TIME'], dtype=float)
        ft = t[np.isfinite(t)]
        if len(ft) > 1:
            duration = ft.max() - ft.min()
            print(f"  TIME span       {ft.min():.4f} to {ft.max():.4f} BJD  "
                  f"(~{duration:.2f} days, ~{int(duration*1440/2):,} 2-min cadences)")


def explain_aperture_hdu(hdu):
    """Explain Extension 2 — the APERTURE image."""
    print("\n" + "=" * 72)
    print("EXTENSION 2 — APERTURE  (the pixel mask used to sum the star's flux)")
    print("=" * 72)
    print("This is a small 2-D image showing which pixels on the CCD were")
    print("summed up to compute the star's brightness for each cadence.")
    print()

    data = hdu.data
    if data is None:
        print("  (No data in this extension.)")
        return

    print(f"  Image shape: {data.shape}  (rows x columns of pixels)")
    print()
    print("  TESS aperture masks are encoded as BIT FLAGS, not plain 0/1/2/3.")
    print("  Each pixel value is a sum of these bits:")
    print("    bit 0  (value   1) -> pixel is in the aperture (collected flux)")
    print("    bit 1  (value   2) -> pixel used for the flux-weighted centroid")
    print("    bit 2  (value   4) -> pixel used for the moment centroid")
    print("    bit 5  (value  32) -> pixel was on the CCD frame")
    print()
    print("  Common values you will see:")
    print("    0   = pixel not in aperture")
    print("    32  = on CCD but NOT in aperture")
    print("    33  = on CCD + in aperture (32 + 1)")
    print("    35  = on CCD + in aperture + flux-weighted centroid (32 + 2 + 1)")
    print("    37  = on CCD + in aperture + moment centroid (32 + 4 + 1)")
    print("    43  = on CCD + in aperture + both centroid methods (32 + 8 + 2 + 1)")

    # Decode every pixel
    in_aperture_mask = (np.asarray(data) & 1).astype(bool)  # bit 0
    n_in_aperture = int(np.sum(in_aperture_mask))
    total_pixels = int(data.size)
    pct = 100.0 * n_in_aperture / total_pixels if total_pixels else 0
    print(f"\n  Pixels in aperture (bit 0 set): {n_in_aperture} / {total_pixels}  ({pct:.1f}%)")
    print(f"  Unique raw values in this HDU: {sorted(set(np.asarray(data).ravel().tolist()))}")

    # ASCII-art preview: '#' = in aperture, '.' = not
    print("\n  Aperture mask (visual):  '#' = pixel in aperture, '.' = not in aperture")
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
# Step 4 — Optional: plot the light curve
# ---------------------------------------------------------------------------

def plot_lightcurve(local_path, tic_id, sector, denoise_method=None):
    """Generate an interactive Plotly chart (opens in browser) or fallback PNG."""
    if _HAS_INTERACTIVE_PLOT:
        # Use the interactive plot module
        with fits.open(local_path) as hdul:
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
        # Fallback: static PNG with matplotlib
        return _plot_lightcurve_png(local_path, tic_id, sector)


def _plot_lightcurve_png(local_path, tic_id, sector):
    """Fallback: save a static PNG plot using matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

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
# Main interactive flow
# ---------------------------------------------------------------------------

CURL_URL_TEMPLATE = (
    "https://archive.stsci.edu/missions/tess/download_scripts/"
    "sector/tesscurl_sector_{sector}_lc.sh"
)


def download_curl_script(sector):
    """Download a TESS sector curl script directly from MAST."""
    url = CURL_URL_TEMPLATE.format(sector=sector)
    local_path = f"tesscurl_sector_{sector}_lc.sh"
    print(f"\nDownloading curl script for Sector {sector} ...")
    print(f"  URL: {url}")
    try:
        urllib.request.urlretrieve(url, local_path)
    except Exception as e:
        print(f"ERROR: could not download the script: {e}")
        print("Check that the sector number is valid (1 to ~104).")
        sys.exit(1)
    size_kb = os.path.getsize(local_path) / 1024
    print(f"  Saved: {local_path}  ({size_kb:.1f} KB)")
    return local_path


def main():
    print("=" * 72)
    print(" TESS Sector Curl File Explorer")
    print("=" * 72)

    # --- Decide how to get the curl script ---
    # CLI shortcut: if a single integer argument is passed, treat it as a sector number
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        script_path = download_curl_script(int(sys.argv[1]))
    elif len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        script_path = sys.argv[1]
    else:
        print("\nHow do you want to provide the curl script?")
        print("  1) Browse for the .sh file on my device  (file picker popup)")
        print("  2) Type the path to the .sh file manually")
        print("  3) Just give me a sector number  (auto-download from MAST)")
        mode = input("Choice [1/2/3]: ").strip()

        if mode == '2':
            script_path = input(
                "Enter the path to the TESS curl script\n"
                "  (e.g. tesscurl_sector_1_lc.sh): "
            ).strip().strip('"').strip("'")

            if not os.path.exists(script_path):
                print(f"\nERROR: file not found: {script_path}")
                print("\nTroubleshooting:")
                print("  - Make sure you're running this script from the same folder")
                print("    as the .sh file, OR give the full path.")
                print("  - On Windows, use forward slashes:  C:/Users/me/Downloads/file.sh")
                print("  - If you don't have the file yet, choose option 3 instead.")
                sys.exit(1)
        elif mode == '3':
            sector_input = input(
                "Enter the sector number (1 to ~104, or 'list' to see all): "
            ).strip()
            if sector_input.lower() == 'list':
                print("Available sectors: 1, 2, 3, ..., 104, 1751 (1751 is a special calibrated FFI sector)")
                sector_input = input("Enter the sector number: ").strip()
            try:
                sector = int(sector_input)
            except ValueError:
                print(f"ERROR: '{sector_input}' is not a valid sector number.")
                sys.exit(1)
            script_path = download_curl_script(sector)
        else:
            # Default / mode 1: file picker
            print("\nOpening file picker dialog...")
            print("(Choose any tesscurl_sector_*.sh file on your Mac and click Open.)")
            script_path = pick_sh_file()
            if not script_path:
                print("\nNo file selected. Exiting.")
                sys.exit(0)
            if not os.path.exists(script_path):
                print(f"\nERROR: file not found: {script_path}")
                sys.exit(1)

    print(f"\nParsing {script_path} ...")
    entries = parse_curl_script(script_path)
    print(f"Found {len(entries):,} stars in this sector.\n")

    if not entries:
        print("No download lines found in the script. Is this really a TESS curl script?")
        sys.exit(1)

    # --- Show preview ---
    print("First 5 stars:")
    for i, e in enumerate(entries[:5]):
        print(f"  [{i+1:>4}] TIC {e['tic_id']}   (sector {e['sector']})")
    if len(entries) > 5:
        print(f"  ... and {len(entries) - 5:,} more")
    print()

    # --- Ask the user how to pick a star ---
    print("How do you want to pick a star?")
    print("  1) Show ALL stars (long list)")
    print("  2) Search by TIC ID")
    print("  3) Pick by line number (1 to N)")
    print("  4) Show stars in a range (e.g. 10-30)")
    choice = input("Choice [1/2/3/4]: ").strip()

    selected = None
    if choice == '1':
        for i, e in enumerate(entries):
            print(f"  [{i+1:>5}] TIC {e['tic_id']}")
        idx = int(input("Enter the number: ")) - 1
        selected = entries[idx]
    elif choice == '2':
        tic = input("Enter TIC ID: ").strip()
        matches = [e for e in entries if e['tic_id'] == tic.zfill(16)]
        if not matches:
            print(f"TIC {tic} not found.")
            sys.exit(1)
        selected = matches[0]
    elif choice == '4':
        rng = input("Enter range (e.g. 10-30): ").strip()
        a, b = map(int, rng.split('-'))
        for i in range(a - 1, b):
            print(f"  [{i+1:>5}] TIC {entries[i]['tic_id']}")
        idx = int(input("Enter the number: ")) - 1
        selected = entries[idx]
    else:
        idx = int(input(f"Enter line number (1-{len(entries)}): ")) - 1
        selected = entries[idx]

    print(f"\nSelected star:")
    print(f"  TIC ID   : {selected['tic_id']}")
    print(f"  Sector   : {selected['sector']}")
    print(f"  Filename : {selected['filename']}")
    print(f"  URL      : {selected['url']}")

    # --- Download if needed ---
    local_path = selected['filename']
    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"\nFile already exists locally: {local_path}  ({size_mb:.2f} MB)")
    else:
        print("\nFile not present locally. Downloading now...")
        download_file(selected['url'], local_path)

    # --- Open the FITS file ---
    print("\n" + "=" * 72)
    print(f"OPENING: {local_path}")
    print("=" * 72)

    with fits.open(local_path) as hdul:
        print(f"Number of HDUs (extensions): {len(hdul)}")
        for i, hdu in enumerate(hdul):
            shape = getattr(hdu.data, 'shape', 'no data')
            print(f"  HDU[{i}]  name={hdu.name!r:<15} data shape={shape}")

        explain_primary_hdu(hdul[0])
        if len(hdul) > 1:
            explain_lightcurve_hdu(hdul[1])
        if len(hdul) > 2:
            explain_aperture_hdu(hdul[2])

    # --- Offer to plot ---
    plot_type = "interactive HTML" if _HAS_INTERACTIVE_PLOT else "PNG"
    answer = input(f"\nGenerate {plot_type} plot of the light curve? [y/N]: ").strip().lower()
    if answer == 'y':
        # Ask which denoising method to apply (only if interactive_plot is available)
        denoise_method = None
        if _HAS_INTERACTIVE_PLOT:
            print("\n  Denoising options:")
            print("    0) None - plot raw PDCSAP_FLUX")
            print("    1) Gaussian Process regression (smooths noise, preserves transits)")
            choice = input("  Choose [0/1, default=0]: ").strip()
            if choice == '1':
                denoise_method = 'gp'

        out = plot_lightcurve(local_path, selected['tic_id'], selected['sector'], denoise_method=denoise_method)
        if out:
            print(f"Plot saved to: {os.path.abspath(out)}")
            if _HAS_INTERACTIVE_PLOT:
                print("(Interactive chart should open in your browser automatically)")

    print("\nDone.")


if __name__ == "__main__":
    main()
