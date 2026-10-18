#!/usr/bin/env python3
"""render_vis.py: convert visualize.cpp PPM dumps into PNGs for the README.

Usage: python3 scripts/render_vis.py <ppm_dir> <out_dir>
Produces: conv1_filters.png, and per-sample digit_<n>_input.png,
digit_<n>_conv1_maps.png, digit_<n>_conv2_maps.png.
"""
import glob
import os
import sys

from PIL import Image

def main() -> None:
    ppm_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(ppm_dir, "*.ppm"))):
        img = Image.open(path)
        dest = os.path.join(out_dir, os.path.basename(path)[:-4] + ".png")
        img.save(dest)
        print(f"{path} -> {dest} ({img.size[0]}x{img.size[1]})")

if __name__ == "__main__":
    main()
