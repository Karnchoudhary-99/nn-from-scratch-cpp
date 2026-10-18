#!/usr/bin/env bash
# Download the four MNIST IDX files into data/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")/../data"
BASE="https://storage.googleapis.com/cvdf-datasets/mnist"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
         t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    if [ -f "$f" ]; then
        echo "$f already present, skipping"
    else
        echo "downloading $f ..."
        curl -sL --max-time 240 -o "$f.gz" "$BASE/$f.gz"
        gzip -dk "$f.gz"
        rm "$f.gz"
    fi
done
echo "done."
