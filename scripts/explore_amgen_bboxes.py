"""
explore_amgen_bboxes.py — Read-only inspection of XIF/AMIDE files in the Amgen SUV data.

Goal: determine whether per-animal bounding box or ROI annotations are embedded in the XIF
files. If found, these can be used instead of CT-based auto-segmentation in build_nifti_dataset.py
to assign individual crop regions to each mouse in a multi-animal scanner session.

AMIDE (Amos Medical Image Data Examiner) stores study data in a directory structure:
  <study>.xif   — top-level file (actually a gzipped tar or a plain text header, version-dependent)
  OR a .xif directory containing a header + raw data files.

Usage:
  python scripts/explore_amgen_bboxes.py [--scan SCAN_ID] [--week WEEK] [--tracer TRACER]
  python scripts/explore_amgen_bboxes.py --scan m54253 --week "Week 12" --tracer FDG
  python scripts/explore_amgen_bboxes.py --summary   # scan all .xif files, report what's found

Reads config.yaml for amgen_data_root path. Never modifies any file.
"""

import os
import sys
import gzip
import tarfile
import struct
import argparse
import yaml


def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file '{config_path}' not found. "
            "Copy config.yaml.example to config.yaml and fill in your paths."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_xif_files(amgen_root, scan_id=None, week=None, tracer=None):
    """
    Walk amgen_data_root and return a list of .xif paths matching the optional filters.
    Filters are matched by substring in the path (case-insensitive for week/tracer).
    """
    matches = []
    for dirpath, dirnames, filenames in os.walk(amgen_root):
        for fname in filenames:
            if not fname.lower().endswith(".xif"):
                continue
            full_path = os.path.join(dirpath, fname)
            rel = os.path.relpath(full_path, amgen_root)

            if scan_id and scan_id not in fname:
                continue
            if week and week.lower() not in rel.lower():
                continue
            if tracer and tracer.lower() not in rel.lower():
                continue
            matches.append(full_path)
    return sorted(matches)


def probe_xif_file(xif_path):
    """
    Inspect a single .xif file to determine its format and extract any bounding box / ROI data.

    Returns a dict with keys:
      format       : "gzip_tar" | "gzip_raw" | "plaintext" | "binary" | "directory" | "unknown"
      header_text  : first 2000 chars of decoded text content (if text-like)
      roi_hints    : list of substrings suggesting ROI/bounding box data
      raw_size_kb  : uncompressed size estimate in KB
      error        : error string if something failed
    """
    result = {
        "path": xif_path,
        "format": "unknown",
        "header_text": "",
        "roi_hints": [],
        "raw_size_kb": 0,
        "error": None,
    }

    if os.path.isdir(xif_path):
        result["format"] = "directory"
        # Walk the directory and list contents
        contents = []
        for root, dirs, files in os.walk(xif_path):
            for f in files:
                contents.append(os.path.relpath(os.path.join(root, f), xif_path))
        result["header_text"] = "\n".join(contents[:50])
        return result

    try:
        file_size = os.path.getsize(xif_path)
        result["raw_size_kb"] = file_size / 1024

        # Try gzip
        try:
            with gzip.open(xif_path, "rb") as gz:
                raw = gz.read(8192)  # read first 8KB only
            # Check if it's a tar inside gzip
            if raw[:5] == b"ustar" or raw[257:262] == b"ustar":
                result["format"] = "gzip_tar"
            else:
                result["format"] = "gzip_raw"
            # Try to decode as text
            try:
                text = raw.decode("utf-8", errors="replace")
                result["header_text"] = text[:2000]
                result["roi_hints"] = _find_roi_hints(text)
            except Exception:
                result["header_text"] = repr(raw[:200])
            return result
        except gzip.BadGzipFile:
            pass
        except Exception as e:
            pass  # Not gzip, try other formats

        # Try tarfile directly
        if tarfile.is_tarfile(xif_path):
            result["format"] = "tar"
            with tarfile.open(xif_path, "r:*") as tf:
                members = tf.getnames()[:20]
                result["header_text"] = "\n".join(members)
            return result

        # Try reading as plain text
        try:
            with open(xif_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(4000)
            result["format"] = "plaintext"
            result["header_text"] = text[:2000]
            result["roi_hints"] = _find_roi_hints(text)
            return result
        except Exception:
            pass

        # Binary fallback — read header bytes
        with open(xif_path, "rb") as f:
            raw = f.read(512)
        result["format"] = "binary"
        result["header_text"] = repr(raw[:128])
        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def _find_roi_hints(text):
    """
    Search text for keywords suggesting ROI, bounding box, or animal annotation data.
    Returns a list of matching keyword occurrences with surrounding context.
    """
    keywords = [
        "roi", "ROI", "bounding", "bbox", "box", "extent",
        "animal", "mouse", "subject", "patient",
        "voi", "VOI", "region", "corner", "min", "max",
        "x_size", "y_size", "z_size", "x_offset", "y_offset", "z_offset",
        "coord", "position", "volume",
    ]
    hints = []
    text_lower = text.lower()
    for kw in keywords:
        idx = text_lower.find(kw.lower())
        if idx >= 0:
            snippet = text[max(0, idx - 20):idx + 60].strip().replace("\n", " ")
            hints.append(f"[{kw}] ...{snippet}...")
    return hints[:20]  # cap at 20 hints


def print_probe_result(result, verbose=False):
    path_short = os.path.basename(os.path.dirname(result["path"])) + "/" + os.path.basename(result["path"])
    print(f"\n{'='*60}")
    print(f"File:   {path_short}")
    print(f"Format: {result['format']}  ({result['raw_size_kb']:.1f} KB on disk)")
    if result["error"]:
        print(f"ERROR:  {result['error']}")
        return
    if result["roi_hints"]:
        print(f"ROI/bbox keywords found ({len(result['roi_hints'])}):")
        for h in result["roi_hints"][:10]:
            print(f"  {h}")
    else:
        print("  No ROI/bbox keywords detected.")
    if verbose and result["header_text"]:
        print(f"\n--- Header preview ---\n{result['header_text'][:800]}\n---")


def summary_mode(amgen_root):
    """
    Walk all .xif files, probe each, and summarise what was found.
    """
    print(f"Scanning: {amgen_root}")
    all_xif = find_xif_files(amgen_root)
    print(f"Found {len(all_xif)} .xif files.\n")

    format_counts = {}
    files_with_roi = []

    for xif_path in all_xif[:200]:  # cap at 200 for safety
        result = probe_xif_file(xif_path)
        fmt = result["format"]
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
        if result["roi_hints"]:
            files_with_roi.append(result["path"])

    print("Format breakdown:")
    for fmt, count in sorted(format_counts.items()):
        print(f"  {fmt}: {count}")

    print(f"\nFiles with ROI/bbox keywords: {len(files_with_roi)}")
    for p in files_with_roi[:20]:
        print(f"  {os.path.relpath(p, amgen_root)}")

    if not files_with_roi:
        print("\n[!] No bounding box / ROI data detected in any XIF file.")
        print("    Fallback: use CT-based auto-segmentation in build_nifti_dataset.py.")
    else:
        print("\n[+] Bounding box / ROI data found — inspect individual files with --scan flag.")


def main():
    parser = argparse.ArgumentParser(description="Inspect Amgen XIF files for bounding box metadata.")
    parser.add_argument("--scan", help="Scan ID to filter (e.g. m54253)")
    parser.add_argument("--week", help="Week folder filter (e.g. 'Week 12')")
    parser.add_argument("--tracer", help="Tracer filter (e.g. FDG or NaF)")
    parser.add_argument("--summary", action="store_true", help="Scan all XIF files and print a summary")
    parser.add_argument("--verbose", action="store_true", help="Print full header preview for each file")
    args = parser.parse_args()

    cfg = load_config()
    amgen_root = cfg["paths"].get("amgen_data_root", "/data1/Amgen SUV Data")

    if not os.path.isdir(amgen_root):
        print(f"[!] amgen_data_root not found: {amgen_root}")
        print("    Set 'paths.amgen_data_root' in config.yaml.")
        sys.exit(1)

    if args.summary:
        summary_mode(amgen_root)
        return

    xif_files = find_xif_files(amgen_root, scan_id=args.scan, week=args.week, tracer=args.tracer)
    if not xif_files:
        print(f"No .xif files found matching filters: scan={args.scan}, week={args.week}, tracer={args.tracer}")
        print(f"  Searched: {amgen_root}")
        sys.exit(1)

    print(f"Found {len(xif_files)} matching .xif file(s).")
    for xif_path in xif_files[:10]:  # inspect up to 10
        result = probe_xif_file(xif_path)
        print_probe_result(result, verbose=args.verbose)

    if len(xif_files) > 10:
        print(f"\n... ({len(xif_files) - 10} more files not shown; use --scan/--week/--tracer to narrow down)")


if __name__ == "__main__":
    main()
