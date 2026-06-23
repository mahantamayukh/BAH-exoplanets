"""
Download TESS SPOC FITS files from MAST using TIC ID

Author: Mayukh
"""

import lightkurve as lk
from pathlib import Path


def download_tess_spoc_fits():

    print("=" * 50)
    print("TESS SPOC FITS Downloader")
    print("=" * 50)

    tic_id = input("\nEnter TIC ID: ").strip()

    target = f"TIC {tic_id}"

    print("\nSearching MAST...")

    try:
        search_result = lk.search_lightcurve(
            target,
            mission="TESS",
            author="SPOC"
        )

        if len(search_result) == 0:
            print("\nNo SPOC light curves found.")
            return

        print("\nAvailable Observations:")
        print(search_result)

        save_dir = Path(f"TIC_{tic_id}_FITS")
        save_dir.mkdir(exist_ok=True)

        print("\nDownloading FITS files...")

        collection = search_result.download_all(
            download_dir=str(save_dir)
        )

        print("\nDownload Complete!")

        print("\nDownloaded Files:")

        for lc in collection:
            try:
                print(lc.filename)
            except:
                pass

        print(f"\nSaved in folder: {save_dir.resolve()}")

    except Exception as e:
        print("\nError:")
        print(e)


if __name__ == "__main__":
    download_tess_spoc_fits()