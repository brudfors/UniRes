#!/usr/bin/env python3
"""Download IXI brain-MRI volumes (T1/T2/PD) for training the DL denoiser prior.

The IXI dataset (https://brain-development.org/ixi-dataset/) ships one big .tar per
modality (~3-5 GB each). This script STREAMS each tar over HTTP and extracts only the
first N files, aborting the connection once N are written -- so `--n 2` downloads only
~tens of MB per modality, which makes it quick to test before pulling the whole set.

Per-contrast pooling: the denoiser is applied per channel, so matched subjects across
contrasts are NOT required -- we just collect a pile of T1, T2 and PD volumes.

Deps: standard library only (urllib, tarfile). nibabel (baked in the image) is used
only for the optional --verify.

Examples:
  python scripts/download_ixi.py --n 2 --out data/ixi --verify     # quick test (2 each)
  python scripts/download_ixi.py --n 0 --out data/ixi              # full download (all)
  python scripts/download_ixi.py --n 50 --modalities T1 T2         # 50 T1 + 50 T2
"""
import argparse
import os
import sys
import tarfile
import time
import urllib.request

# The original IXI server (biomedic.doc.ic.ac.uk) currently 403s its whole downloads
# directory, so default to the Hugging Face mirror that hosts the SAME IXI-<mod>.tar
# files. Both are streamable; pick with --source.
SOURCES = {
    "hf": "https://huggingface.co/datasets/Santhosh1884/IXI-Datasets/resolve/main",
    "biomedic": "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI",
}
TARS = {"T1": "IXI-T1.tar", "T2": "IXI-T2.tar", "PD": "IXI-PD.tar", "MRA": "IXI-MRA.tar"}


def stream_extract(modality, base, out_root, n, timeout, retries=3):
    """Stream IXI-<modality>.tar and extract up to n .nii.gz files (n<=0 => all)."""
    url = f"{base}/{TARS[modality]}"
    dst = os.path.join(out_root, modality)
    os.makedirs(dst, exist_ok=True)
    limit = None if (n is None or n <= 0) else n
    print(f"[{modality}] streaming {url}  (target: {'all' if limit is None else limit} files)")

    for attempt in range(1, retries + 1):
        got, t0 = [], time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "unires-ixi-dl/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # Uncompressed .tar served over HTTP -> streaming (non-seekable) read.
                with tarfile.open(fileobj=resp, mode="r|") as tar:
                    for member in tar:
                        if not member.isfile():
                            continue
                        name = os.path.basename(member.name)
                        if not name.endswith(".nii.gz"):
                            continue
                        out_path = os.path.join(dst, name)
                        data = tar.extractfile(member).read()   # must read before advancing
                        with open(out_path, "wb") as w:
                            w.write(data)
                        got.append(name)
                        print(f"  [{len(got):>3}] {name}  ({len(data) // 1024} KB)")
                        if limit is not None and len(got) >= limit:
                            break
            dt = time.time() - t0
            print(f"[{modality}] done: {len(got)} file(s) -> {dst}  ({dt:.1f}s)")
            return got
        except Exception as e:  # network hiccup / partial stream -> retry
            print(f"[{modality}] attempt {attempt}/{retries} failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def verify(out_root, modalities):
    try:
        import nibabel as nib
    except ImportError:
        print("--verify skipped: nibabel not available", file=sys.stderr)
        return
    print("\n=== verify (nibabel) ===")
    ok = bad = 0
    for m in modalities:
        d = os.path.join(out_root, m)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if not f.endswith(".nii.gz"):
                continue
            p = os.path.join(d, f)
            try:
                img = nib.load(p)
                print(f"  {m}/{f}: shape {img.shape}, vox {tuple(round(float(z),2) for z in img.header.get_zooms()[:3])}")
                ok += 1
            except Exception as e:
                print(f"  {m}/{f}: LOAD FAILED -- {e}", file=sys.stderr)
                bad += 1
    print(f"verify: {ok} ok, {bad} bad")


def main():
    ap = argparse.ArgumentParser(description="Download IXI T1/T2/PD volumes (streamed).")
    ap.add_argument("--out", default="data/ixi", help="output root (default: data/ixi)")
    ap.add_argument("--source", default="hf", choices=list(SOURCES.keys()),
                    help="download source (default: hf mirror; biomedic often 403s)")
    ap.add_argument("--modalities", nargs="+", default=["T1", "T2", "PD"],
                    choices=list(TARS.keys()), help="modalities to fetch")
    ap.add_argument("--n", type=int, default=2,
                    help="files per modality; <=0 means ALL (default: 2 for a quick test)")
    ap.add_argument("--timeout", type=int, default=60, help="socket timeout (s)")
    ap.add_argument("--verify", action="store_true", help="load each file with nibabel")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base = SOURCES[args.source]
    print(f"source: {args.source} ({base})")
    total = 0
    for m in args.modalities:
        total += len(stream_extract(m, base, args.out, args.n, args.timeout))
    print(f"\nTOTAL: {total} file(s) under {args.out}/")
    if args.verify:
        verify(args.out, args.modalities)


if __name__ == "__main__":
    main()
