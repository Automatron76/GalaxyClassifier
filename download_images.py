"""
Step 2 — Download SDSS galaxy cutout images.

For each galaxy in the labels CSV, this script requests a 256×256 px
JPEG cutout from the SDSS SkyServer DR14 Image Cutout API using the
galaxy's RA/Dec coordinates.

Usage
-----
    python download_images.py              # default: first 5 000 galaxies
    python download_images.py --limit 500  # download only 500

Notes
-----
* Images are saved as  data/processed/train/{galaxy_id}.jpg
* Already-downloaded images are skipped (safe to re-run).
* A 0.3 s delay is inserted between requests to avoid overloading
  the SDSS server (rate limiting).
"""

import argparse
import os

import pandas as pd
import requests
from time import sleep

from config import LABELS_PATH, IMAGES_DIR

# SDSS SkyServer Image Cutout endpoint (Data Release 14)
SDSS_CUTOUT_URL = (
    "https://skyserver.sdss.org/dr14/SkyServerWS/ImgCutout/getjpeg"
)

# Image parameters
SCALE = 0.2        # arcseconds per pixel
WIDTH = 256        # pixels
HEIGHT = 256       # pixels

# Pause between HTTP requests to respect the server (seconds)
REQUEST_DELAY = 0.3


def download_galaxy_image(objid: str, ra: float, dec: float,
                          output_dir: str) -> bool:
    """Download a single galaxy cutout and save it as JPEG.

    Returns True on success, False on failure.
    """
    out_path = os.path.join(output_dir, f"{objid}.jpg")

    # Skip if already downloaded
    if os.path.exists(out_path):
        return True

    url = (
        f"{SDSS_CUTOUT_URL}"
        f"?ra={ra}&dec={dec}&scale={SCALE}"
        f"&width={WIDTH}&height={HEIGHT}"
    )

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and response.content:
            with open(out_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"  [WARN] HTTP {response.status_code} for {objid}")
            return False
    except Exception as e:
        print(f"  [ERROR] {objid}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download SDSS galaxy images")
    parser.add_argument(
        "--limit", type=int, default=5000,
        help="Max number of galaxies to download (default: 5000)",
    )
    args = parser.parse_args()

    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Read galaxy IDs as strings to preserve exact 18-digit numbers
    df = pd.read_csv(LABELS_PATH, dtype={"id": "string"}).head(args.limit)

    downloaded = 0
    failed = 0

    print(f"Downloading up to {len(df)} galaxy images → {IMAGES_DIR}/\n")

    for i, row in df.iterrows():
        objid = row["id"]
        ra = float(row["ra"])
        dec = float(row["dec"])

        ok = download_galaxy_image(objid, ra, dec, IMAGES_DIR)

        if ok:
            downloaded += 1
        else:
            failed += 1

        # Print progress every 100 images
        total = downloaded + failed
        if total % 100 == 0:
            print(f"  Progress: {total} / {len(df)}  "
                  f"(ok={downloaded}, failed={failed})")

        sleep(REQUEST_DELAY)

    print(f"\nDone. Downloaded: {downloaded}, failed: {failed}")


if __name__ == "__main__":
    main()
