"""Offline indexing: process videos once so later searches are instant.

    python index_videos.py                     # every video in data/videos
    python index_videos.py path/to/video.mp4   # one or more specific files
    python index_videos.py --max-seconds 60    # only the first minute of each (quick test)

Camera names default to the file name. Use name=camera to set one, e.g.
    python index_videos.py data/videos/gate.mp4=Gate-Cam
"""
import argparse
import sys
import time
from pathlib import Path

from core import store
from core.config import GLOBAL_ID_THRESHOLD, VIDEO_EXTS, VIDEOS_DIR, to_abs
from core.detect_track import load_detector, process_video
from core.reid import ReIDModel
from core.search import assign_global_ids
from core.text_search import ClipModel


def load_models():
    return {"detector": load_detector(), "reid": ReIDModel(), "clip": ClipModel()}


def index_one(models, path, camera=None, max_seconds=0, progress=None):
    meta, tracks, boxes, reid_emb, clip_emb = process_video(
        path, models["detector"], models["reid"], models["clip"], camera=camera, max_seconds=max_seconds,
        progress=progress)
    con = store.connect()
    store.add_video(con, meta, tracks, boxes, reid_emb, clip_emb)
    con.close()
    return meta, tracks


def refresh_global_ids(threshold=GLOBAL_ID_THRESHOLD):
    index = store.load_index()
    mapping = assign_global_ids(index, threshold)
    store.set_global_ids(mapping)
    return len(set(mapping.values()))


def refresh_attributes():
    """Recompute clothing colours from the stored crops (after changing the colour rules)."""
    import json
    from collections import Counter

    import cv2
    import pandas as pd

    from core.text_search import crop_colours
    con = store.connect()
    samples = pd.read_sql("SELECT track_pk, crop_path FROM samples", con)
    for pk, g in samples.groupby("track_pk"):
        ups, los = Counter(), Counter()
        for path in g["crop_path"]:
            img = cv2.imread(to_abs(path))
            if img is None:
                continue
            up, lo = crop_colours(img)
            ups.update(up[:1]); los.update(lo[:1])
        con.execute("UPDATE tracks SET upper_colours=?, lower_colours=? WHERE id=?",
                    (json.dumps([c for c, _ in ups.most_common(2)]), json.dumps([c for c, _ in los.most_common(2)]),
                     int(pk)))
    con.commit()
    con.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*", help="video files (default: everything in data/videos)")
    ap.add_argument("--max-seconds", type=float, default=0, help="only index the first N seconds of each video")
    ap.add_argument("--refresh-attributes", action="store_true", help="only recompute clothing colours")
    ap.add_argument("--refresh-global-ids", action="store_true", help="only recompute Global IDs")
    args = ap.parse_args()
    if args.refresh_attributes or args.refresh_global_ids:
        if args.refresh_attributes:
            refresh_attributes()
            print("Clothing colours recomputed.")
        if args.refresh_global_ids:
            print(f"{refresh_global_ids()} Global IDs assigned.")
        return

    items = []
    for v in args.videos or sorted(str(p) for p in VIDEOS_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS):
        path, _, cam = v.partition("=")
        items.append((Path(path), cam or None))
    if not items:
        sys.exit(f"No videos found. Put .mp4/.avi files in {VIDEOS_DIR}")

    print("Loading models ...")
    models = load_models()
    for path, cam in items:
        print(f"\n== {path.name}")

        def progress(done, total):
            print(f"\r   frame {done}/{total}", end="", flush=True)

        t0 = time.time()
        meta, tracks = index_one(models, path, cam, args.max_seconds, progress)
        secs = time.time() - t0
        print(f"\n   {len(tracks)} person tracks, {sum(len(t['samples']) for t in tracks)} crops embedded, "
              f"{meta['frames']} frames in {secs:.1f}s ({meta['frames'] / max(secs, 1e-6):.1f} FPS)")

    n = refresh_global_ids()
    print(f"\nAssigned {n} Global IDs across all indexed videos. Index saved in data/index/")


if __name__ == "__main__":
    main()
