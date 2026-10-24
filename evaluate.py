"""Evaluate detection, tracking, re-identification, search and speed.

    python evaluate.py            # everything (needs indexed WILDTRACK videos + eval/wildtrack_gt.csv)
    python evaluate.py --no-coco  # skip the COCO detection benchmark (needs internet the first time)

Writes eval/results.json (shown on the app's Evaluation page) and eval/*.png figures.
"""
import argparse
import json
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

from core import store  # noqa: E402
from core.config import EVAL_DIR, YOLO_WEIGHTS, get_device  # noqa: E402
from core.search import search_by_reid_vector, search_text, track_means  # noqa: E402

GT_CSV = EVAL_DIR / "wildtrack_gt.csv"
TEXT_QUERIES = EVAL_DIR / "text_queries.json"


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def annotated_regions(gt):
    """WILDTRACK only labels people standing inside a fixed ground area. Detections outside it
    are real people but unlabelled, so they are ignored when counting false positives."""
    regions = {}
    for video, g in gt.groupby("video"):
        feet = np.c_[(g["x1"] + g["x2"]) / 2, g["y2"]]
        hull = ConvexHull(feet)
        regions[video] = MplPath(feet[hull.vertices])
    return regions


def in_region(region, boxes):
    feet = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
    return region.contains_points(feet, radius=15)


# ---------------------------------------------------------------------------

def eval_detection_tracking(index, gt, regions):
    import motmetrics as mm
    con = store.connect()
    boxes = pd.read_sql("SELECT * FROM boxes", con)
    con.close()
    vids = index["videos"].set_index("id")["name"]
    boxes["video"] = boxes["video_id"].map(vids)
    tp = fp = fn = 0
    accs, names = [], []
    for video, g in gt.groupby("video"):
        b = boxes[boxes["video"] == video]
        if b.empty:
            continue
        acc = mm.MOTAccumulator(auto_id=True)
        region = regions[video]
        for frame in sorted(set(g["frame"]) | set(b["frame"])):
            gf = g[g["frame"] == frame]
            bf = b[b["frame"] == frame]
            bb = bf[["x1", "y1", "x2", "y2"]].to_numpy()
            keep = in_region(region, bb) if len(bb) else np.zeros(0, bool)
            bf, bb = bf[keep], bb[keep]
            gb = gf[["x1", "y1", "x2", "y2"]].to_numpy()
            ious = iou_matrix(gb, bb) if len(gb) and len(bb) else np.zeros((len(gb), len(bb)))
            dist = np.where(ious >= 0.5, 1 - ious, np.nan)
            acc.update(gf["person_id"].tolist(), bf["track_id"].tolist(), dist)
            # one-to-one matching (Hungarian) for precision / recall
            matched = 0
            if ious.size:
                ri, ci = linear_sum_assignment(-ious)
                matched = int((ious[ri, ci] >= 0.5).sum())
            tp += matched; fp += len(bb) - matched; fn += len(gb) - matched
        accs.append(acc); names.append(video)
    mh = mm.metrics.create()
    summ = mh.compute_many(accs, names=names, metrics=["mota", "idf1", "num_switches", "num_fragmentations",
                                                       "mostly_tracked", "num_unique_objects"],
                           generate_overall=True)
    o = summ.loc["OVERALL"]
    det = {"Precision": f"{tp / max(tp + fp, 1):.3f}", "Recall": f"{tp / max(tp + fn, 1):.3f}",
           "F1": f"{2 * tp / max(2 * tp + fp + fn, 1):.3f}", "GT boxes": f"{tp + fn:,}"}
    trk = {"MOTA": f"{o['mota']:.3f}", "IDF1": f"{o['idf1']:.3f}", "ID switches": int(o["num_switches"]),
           "Fragmentations": int(o["num_fragmentations"]),
           "Mostly tracked": f"{int(o['mostly_tracked'])}/{int(o['num_unique_objects'])}"}
    return det, trk


def label_tracks(index, gt):
    """Give each indexed track the ground-truth person ID its crops overlap most (IoU >= 0.5)."""
    samples = index["samples"]
    vids = index["videos"].set_index("id")["name"]
    gt_by = {k: g for k, g in gt.groupby(["video", "frame"])}
    sample_pid = np.full(len(samples), -1)
    for i, s in enumerate(samples.itertuples()):
        g = gt_by.get((vids[s.video_id], s.frame))
        if g is None:
            continue
        ious = iou_matrix([[s.x1, s.y1, s.x2, s.y2]], g[["x1", "y1", "x2", "y2"]].to_numpy())[0]
        if ious.max() >= 0.5:
            sample_pid[i] = g["person_id"].iloc[int(ious.argmax())]
    samples = samples.assign(pid=sample_pid)
    labels = {}
    for pk, g in samples.groupby("track_pk"):
        c = Counter(p for p in g["pid"] if p >= 0)
        if c:
            pid, n = c.most_common(1)[0]
            if n >= 0.5 * len(g):
                labels[pk] = pid
    return labels


def eval_reid(index, labels):
    tracks = index["tracks"].set_index("id")
    pks, feats = track_means(index)
    pos = {pk: i for i, pk in enumerate(pks)}
    lab = [pk for pk in pks if pk in labels]
    vid = tracks["video_id"]

    r1 = r5 = 0
    aps, n_q = [], 0
    same, diff = [], []
    for q in lab:
        gal = [g for g in lab if vid[g] != vid[q]]
        if not any(labels[g] == labels[q] for g in gal):
            continue
        sims = feats[[pos[g] for g in gal]] @ feats[pos[q]]
        order = np.argsort(-sims)
        hits = np.array([labels[gal[i]] == labels[q] for i in order])
        n_q += 1
        r1 += hits[0]; r5 += hits[:5].any()
        prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
        aps.append((prec * hits).sum() / hits.sum())
        for g, s in zip(gal, sims):
            (same if labels[g] == labels[q] else diff).append(float(s))
    track_level = {"Queries": n_q, "Rank-1": f"{r1 / max(n_q, 1):.3f}", "Rank-5": f"{r5 / max(n_q, 1):.3f}",
                   "mAP": f"{np.mean(aps):.3f}" if aps else "n/a"}
    return track_level, np.array(same), np.array(diff)


def eval_image_search(index, labels):
    """Photo search: one crop of a person from one camera, searched against the other cameras."""
    tracks = index["tracks"].set_index("id")
    samples = index["samples"]
    lab_by_video = {}
    for pk, pid in labels.items():
        lab_by_video.setdefault(pid, set()).add(tracks.loc[pk, "video_id"])
    r1 = r5 = 0
    n, times = 0, []
    for pk, pid in labels.items():
        if len(lab_by_video[pid]) < 2:
            continue
        s = samples[samples["track_pk"] == pk]
        best = s.loc[(s["y2"] - s["y1"]).idxmax()]
        q = index["reid"][int(best["id"])]
        other = tracks[tracks["video_id"] == tracks.loc[pk, "video_id"]].index
        t0 = time.perf_counter()
        df = search_by_reid_vector(index, q, top_k=5, exclude_track_pks=set(other))
        times.append(time.perf_counter() - t0)
        hits = [labels.get(t) == pid for t in df["id"]]
        n += 1; r1 += bool(hits and hits[0]); r5 += any(hits)
    return {"Queries": n, "Top-1": f"{r1 / max(n, 1):.3f}", "Top-5": f"{r5 / max(n, 1):.3f}",
            "Avg latency (ms)": f"{1000 * np.mean(times):.1f}" if times else "n/a"}


def eval_photo_queries(index, labels):
    """Photos taken by WILDTRACK cameras that were NOT indexed (eval/query_photos)."""
    qfile = EVAL_DIR / "query_photos" / "queries.json"
    if not qfile.exists():
        return None
    import cv2
    from core.detect_track import load_detector
    from core.reid import ReIDModel
    from core.search import search_image
    models = {"detector": load_detector(), "reid": ReIDModel()}
    r1 = r5 = r10 = 0
    times = []
    queries = [q for q in json.loads(qfile.read_text()) if q["person_id"] in set(labels.values())]
    for q in queries:
        img = cv2.imread(str(EVAL_DIR / "query_photos" / q["file"]))
        df, _, secs = search_image(index, models, img, top_k=10)
        times.append(secs)
        hits = [labels.get(t) == q["person_id"] for t in df["id"]]
        r1 += hits[0]; r5 += any(hits[:5]); r10 += any(hits)
    n = max(len(queries), 1)
    return {"Queries": len(queries), "Top-1": f"{r1 / n:.3f}", "Top-5": f"{r5 / n:.3f}", "Top-10": f"{r10 / n:.3f}",
            "Avg latency incl. detection (ms)": f"{1000 * np.mean(times):.0f}"}


def eval_text_search(index, labels):
    if not TEXT_QUERIES.exists():
        return None
    from core.text_search import ClipModel
    models = {"clip": ClipModel()}
    known = set(labels.values())
    queries = [q for q in json.loads(TEXT_QUERIES.read_text()) if q["person_id"] in known]
    r1 = r5 = r10 = 0
    times = []
    for q in queries:
        df, _, secs = search_text(index, models, q["text"], top_k=10)
        times.append(secs)
        hits = [labels.get(t) == q["person_id"] for t in df["id"]]
        r1 += hits[0]; r5 += any(hits[:5]); r10 += any(hits[:10])
    n = len(queries)
    return {"Queries": n, "Top-1": f"{r1 / n:.3f}", "Top-5": f"{r5 / n:.3f}", "Top-10": f"{r10 / n:.3f}",
            "Avg latency (ms)": f"{1000 * np.mean(times):.0f}"}


def eval_global_ids(index, labels):
    """Pairwise precision/recall of the automatic Global ID grouping."""
    gids = index["tracks"].set_index("id")["global_id"]
    pk = list(labels)
    tp = fp = fn = 0
    for i in range(len(pk)):
        for j in range(i + 1, len(pk)):
            same_gt = labels[pk[i]] == labels[pk[j]]
            same_pred = gids[pk[i]] == gids[pk[j]]
            tp += same_gt and same_pred; fp += (not same_gt) and same_pred; fn += same_gt and not same_pred
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"Pairwise precision": f"{p:.3f}", "Pairwise recall": f"{r:.3f}",
            "Global IDs": int(index["tracks"]["global_id"].nunique()), "GT people": len(set(labels.values()))}


def eval_coco():
    from ultralytics import YOLO
    m = YOLO(str(YOLO_WEIGHTS))
    r = m.val(data="coco128.yaml", classes=[0], imgsz=640, device=get_device(), verbose=False, plots=False)
    speed = r.speed.get("inference", 0)
    return {"Precision": f"{r.box.mp:.3f}", "Recall": f"{r.box.mr:.3f}", "mAP@0.5": f"{r.box.map50:.3f}",
            "mAP@0.5:0.95": f"{r.box.map:.3f}", "Inference (ms/img)": f"{speed:.1f}",
            "FPS (model only)": f"{1000 / speed:.1f}" if speed else "n/a"}


def histogram(same, diff, path):
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=120)
    bins = np.linspace(min(diff.min(), same.min()), 1, 50)
    ax.hist(diff, bins=bins, alpha=0.6, density=True, label=f"different people (n={len(diff):,})", color="#8a8f98")
    ax.hist(same, bins=bins, alpha=0.7, density=True, label=f"same person (n={len(same):,})", color="#2a7de1")
    ax.set_xlabel("Re-ID cosine similarity between tracks in different cameras")
    ax.set_ylabel("density"); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-coco", action="store_true")
    args = ap.parse_args()

    index = store.load_index()
    gt = pd.read_csv(GT_CSV)
    n_frames = index["videos"].set_index("name")["frames"]
    gt = gt[gt["video"].isin(set(n_frames.index))]
    gt = gt[gt["frame"] < gt["video"].map(n_frames)]  # only the part of each video that was indexed
    regions = annotated_regions(gt)
    sections, notes = {}, []

    print("Detection + tracking vs WILDTRACK ground truth ...")
    det, trk = eval_detection_tracking(index, gt, regions)
    sections["Detection on WILDTRACK (IoU ≥ 0.5)"] = det
    sections["Tracking on WILDTRACK"] = trk
    if not args.no_coco:
        print("Detection on COCO128 (person class) ...")
        sections["Detection on COCO128 (person class, standard benchmark)"] = eval_coco()

    print("Re-identification ...")
    labels = label_tracks(index, gt)
    reid, same, diff = eval_reid(index, labels)
    sections["Cross-camera Re-ID (track query vs other cameras)"] = reid
    sections["Image search (single photo vs other cameras)"] = eval_image_search(index, labels)
    photo = eval_photo_queries(index, labels)
    if photo:
        sections["Photo search from cameras that were never indexed (C2, C4, C7)"] = photo
    txt = eval_text_search(index, labels)
    if txt:
        sections["Description search"] = txt
    sections["Global ID grouping"] = eval_global_ids(index, labels)
    if len(same) and len(diff):
        histogram(same, diff, EVAL_DIR / "similarity_hist.png")
        thr = np.linspace(0, 1, 101)
        tpr = [(same >= t).mean() for t in thr]; fpr = [(diff >= t).mean() for t in thr]
        best = thr[int(np.argmax(np.array(tpr) - np.array(fpr)))]
        sections["Similarity distribution"] = {
            "Same-person mean": f"{same.mean():.3f}", "Different-person mean": f"{diff.mean():.3f}",
            "Best separating threshold": f"{best:.2f}",
            "TPR at that threshold": f"{(same >= best).mean():.3f}", "FPR at that threshold": f"{(diff >= best).mean():.3f}"}

    v = index["videos"]
    sections["Speed (indexing, this machine)"] = {
        "Videos": len(v), "Total video length (s)": f"{v['duration_s'].sum():.0f}",
        "Total indexing time (s)": f"{v['processing_s'].sum():.0f}",
        "Frames / second": f"{v['frames'].sum() / v['processing_s'].sum():.2f}",
        "Stored person crops": f"{len(index['samples']):,}", "Tracks": len(index["tracks"])}

    notes += [
        "WILDTRACK: 3 of the 7 synchronised cameras (C1, C3, C5), 400 frames each at 2 frames/second (200 s), "
        "1280×720. Ground-truth person IDs are shared across cameras.",
        "WILDTRACK only labels people inside a fixed ground area; detections whose feet fall outside that area are "
        "ignored rather than counted as false positives.",
        "Tracking runs at 2 frames/second here (the dataset's annotation rate), which is much harder than normal "
        "25–30 FPS video: people move far between frames, so expect more ID switches than on regular footage.",
        "Re-ID uses OSNet-AIN trained on MSMT17 only; it never saw WILDTRACK, so these are cross-domain numbers.",
        "Scores are appearance similarity and are evidence for a human reviewer, not proof of identity.",
    ]
    res = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "device": get_device(),
           "dataset": "WILDTRACK multi-camera dataset (EPFL) + COCO128", "sections": sections,
           "figures": ["similarity_hist.png"], "notes": notes}
    (EVAL_DIR / "results.json").write_text(json.dumps(res, indent=2))
    for name, vals in sections.items():
        print(f"\n{name}")
        for k, val in (vals.items() if isinstance(vals, dict) else []):
            print(f"  {k:28s} {val}")
    print("\nSaved eval/results.json")


if __name__ == "__main__":
    main()
