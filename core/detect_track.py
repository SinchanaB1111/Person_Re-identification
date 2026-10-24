"""Offline video processing: detect -> track -> crop -> embed.

Frames are analysed at about PROCESS_FPS frames per second (every frame for low-frame-rate
CCTV). That is plenty for people walking and keeps CPU indexing time reasonable.

For every video this produces the person tracks (local track IDs within that video),
a few sampled crops per track with Re-ID and CLIP embeddings, the per-frame boxes used
later to highlight a person in a clip, and an annotated H.264 video.
"""
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .config import (ANNOTATED_DIR, ANNOTATED_MAX_WIDTH, BAG_CLASSES, CROPS_DIR, DET_CONF, DET_IMGSZ,
                     MIN_CROP_HEIGHT, MIN_TRACK_SAMPLES, PERSON, PROCESS_FPS, SAMPLE_INTERVAL_S, YOLO_WEIGHTS,
                     get_device)
from .text_search import crop_colours
from .tracker import AppearanceTracker
from .video import H264Writer, draw_box, fit_width, format_ts

_PALETTE = [(56, 168, 0), (255, 112, 31), (0, 194, 255), (182, 0, 255), (0, 120, 255), (147, 69, 52),
            (255, 56, 132), (52, 147, 26), (199, 55, 255), (31, 112, 255), (100, 115, 255), (0, 24, 236)]


def load_detector(weights=YOLO_WEIGHTS):
    return YOLO(str(weights))


def _clip_box(box, w, h):
    x1, y1, x2, y2 = box
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _bags_for_person(pbox, bag_boxes):
    """Bags whose centre lies inside (a slightly widened) person box belong to that person."""
    x1, y1, x2, y2 = pbox
    pad = 0.25 * (x2 - x1)
    found = set()
    for name, (bx1, by1, bx2, by2) in bag_boxes:
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        if x1 - pad <= cx <= x2 + pad and y1 <= cy <= y2:
            found.add(name)
    return found


def process_video(video_path, detector, reid_model, clip_model, camera=None, conf=DET_CONF, imgsz=DET_IMGSZ,
                  sample_interval_s=SAMPLE_INTERVAL_S, max_seconds=0, progress=None):
    """Run the full indexing pipeline on one video.

    Returns (video_meta, tracks, boxes, reid_emb, clip_emb). `progress(done, total)` is
    called every few frames so the UI can show a progress bar.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(max_seconds * fps) if max_seconds else 0
    if max_frames:
        total = min(total, max_frames) if total else max_frames
    sample_every = max(1, round(fps * sample_interval_s))

    stem = video_path.stem
    crop_dir = CROPS_DIR / stem
    crop_dir.mkdir(parents=True, exist_ok=True)
    for old in crop_dir.glob("*.jpg"):
        old.unlink()
    annotated_path = ANNOTATED_DIR / f"{stem}_annotated.mp4"
    writer = None

    stride = max(1, round(fps / PROCESS_FPS))
    device = get_device()
    tracker = AppearanceTracker()
    info = defaultdict(lambda: {"frames": [], "samples": [], "bags": Counter(), "upper": [], "lower": []})
    boxes = []
    sample_crops, sample_reid = [], []
    drawn = []  # boxes drawn on the frames between two analysed frames
    start = time.time()
    frame_idx = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break

            if frame_idx % stride == 0:
                res = detector.predict(frame, classes=[PERSON, *BAG_CLASSES], conf=conf, imgsz=imgsz,
                                       device=device, verbose=False)[0]
                pboxes, pconfs, bag_boxes = [], [], []
                if res.boxes is not None and len(res.boxes):
                    for b, c, cf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int),
                                        res.boxes.conf.cpu().numpy()):
                        if c in BAG_CLASSES:
                            bag_boxes.append((BAG_CLASSES[c], b))
                        elif c == PERSON:
                            pboxes.append(_clip_box(b, width, height))
                            pconfs.append(float(cf))

                def crop_of(i):
                    x1, y1, x2, y2 = [int(v) for v in pboxes[i]]
                    return frame[y1:y2, x1:x2]

                def embed(indices):
                    indices = [i for i in indices if crop_of(i).size]
                    vecs = reid_model.extract([crop_of(i) for i in indices])
                    return dict(zip(indices, vecs))

                assigned, feats = tracker.update(frame_idx / fps, pboxes, pconfs, embed)
                drawn = []
                for di, tid in assigned.items():
                    box, cf = pboxes[di], pconfs[di]
                    t = info[tid]
                    t["frames"].append(frame_idx)
                    boxes.append((tid, frame_idx, *[float(v) for v in box]))
                    t["bags"].update(_bags_for_person(box, bag_boxes))
                    x1, y1, x2, y2 = [int(v) for v in box]
                    take = (not t["samples"] or frame_idx - t["samples"][-1]["frame"] >= sample_every)
                    if take and (y2 - y1) >= MIN_CROP_HEIGHT and (x2 - x1) >= MIN_CROP_HEIGHT // 4:
                        crop = crop_of(di).copy()
                        emb = feats.get(di)
                        if emb is None:
                            emb = reid_model.extract([crop])[0]
                            tracker.add_feature(tid, emb)
                        up, lo = crop_colours(crop)
                        t["upper"].append(tuple(up))
                        t["lower"].append(tuple(lo))
                        crop_path = crop_dir / f"t{tid}_f{frame_idx}.jpg"
                        cv2.imwrite(str(crop_path), crop)
                        t["samples"].append({"frame": frame_idx, "ts": frame_idx / fps, "box": [x1, y1, x2, y2],
                                             "conf": cf, "crop": str(crop_path), "row": len(sample_crops)})
                        sample_crops.append(crop)
                        sample_reid.append(emb)
                    drawn.append((box, tid))

            vis = frame.copy()
            for box, tid in drawn:
                draw_box(vis, box, f"ID {tid}", _PALETTE[tid % len(_PALETTE)], 2)
            cv2.putText(vis, format_ts(frame_idx / fps), (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)
            vis, _ = fit_width(vis, ANNOTATED_MAX_WIDTH)
            if writer is None:
                writer = H264Writer(annotated_path, fps, (vis.shape[1], vis.shape[0]))
            writer.write(vis)

            if progress and frame_idx % 5 == 0:
                progress(frame_idx + 1, total)
    finally:
        cap.release()
        if writer:
            writer.close()

    # CLIP embeddings for all sampled crops in batches (much faster than one at a time).
    reid_emb = np.stack(sample_reid).astype(np.float32) if sample_reid else np.zeros((0, 512), np.float32)
    clip_emb = clip_model.encode_images(sample_crops)

    tracks, keep_rows = [], []
    for tid, t in sorted(info.items()):
        if len(t["samples"]) < MIN_TRACK_SAMPLES:
            continue
        n = len(t["samples"])
        upper = [c for c, _ in Counter(c for u in t["upper"] for c in u[:1]).most_common(2)]
        lower = [c for c, _ in Counter(c for u in t["lower"] for c in u[:1]).most_common(2)]
        bags = sorted(b for b, k in t["bags"].items() if k >= max(2, 0.25 * len(t["frames"])))
        rep = max(t["samples"], key=lambda s: (s["box"][3] - s["box"][1]) * s["conf"])
        for s in t["samples"]:
            keep_rows.append(s["row"])
        tracks.append({"track_id": tid, "first_frame": t["frames"][0], "last_frame": t["frames"][-1],
                       "first_ts": t["frames"][0] / fps, "last_ts": t["frames"][-1] / fps,
                       "detections": len(t["frames"]), "samples": t["samples"], "upper": upper,
                       "lower": lower, "bags": bags, "rep_crop": rep["crop"], "n": n})

    # Re-number the kept samples so rows line up with the trimmed embedding matrices.
    remap = {old: new for new, old in enumerate(keep_rows)}
    for t in tracks:
        for s in t["samples"]:
            s["row"] = remap[s["row"]]
    keep_tids = {t["track_id"] for t in tracks}
    boxes = [b for b in boxes if b[0] in keep_tids]
    reid_emb = reid_emb[keep_rows] if keep_rows else reid_emb[:0]
    clip_emb = clip_emb[keep_rows] if keep_rows else clip_emb[:0]

    frames_done = frame_idx + 1
    meta = {"name": video_path.name, "camera": camera or stem, "path": str(video_path), "fps": fps,
            "frames": frames_done, "width": width, "height": height, "duration_s": frames_done / fps,
            "annotated_path": str(annotated_path), "processing_s": round(time.time() - start, 1),
            "stride": stride}
    if progress:
        progress(total or frames_done, total or frames_done)
    return meta, tracks, boxes, reid_emb, clip_emb
