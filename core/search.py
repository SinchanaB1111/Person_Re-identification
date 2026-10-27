"""Online search over the pre-computed index (no video is re-processed at query time).

All searches score individual sampled crops, then roll scores up to tracks: a track's
score is the mean of its best 3 crop scores, which is robust to one bad crop.
"""
import time

import numpy as np
import pandas as pd

from .config import GLOBAL_ID_THRESHOLD, PERSON, get_device
from .text_search import attribute_bonus, parse_description

TOP_SAMPLES = 3


def _track_scores(index, sample_scores):
    """sample_scores: array aligned with index['samples'] -> DataFrame of tracks with 'score'."""
    s = index["samples"][["id", "track_pk"]].copy()
    s["score"] = sample_scores[s["id"].to_numpy()]
    best = (s.sort_values("score", ascending=False).groupby("track_pk").head(TOP_SAMPLES)
             .groupby("track_pk")["score"].mean())
    best_frame = s.loc[s.groupby("track_pk")["score"].idxmax(), ["track_pk", "id"]].set_index("track_pk")["id"]
    out = index["tracks"].set_index("id").join(best.rename("score"), how="inner")
    out["best_sample"] = best_frame.reindex(out.index).to_numpy()
    out = out.reset_index().rename(columns={"index": "id"})
    return out


def _decorate(index, df):
    """Add video name, camera, best-matching frame/time/crop to a ranked track table."""
    videos = index["videos"].set_index("id")
    samples = index["samples"].set_index("id")
    df = df.copy()
    df["video"] = df["video_id"].map(videos["name"])
    df["camera"] = df["video_id"].map(videos["camera"])
    df["video_path"] = df["video_id"].map(videos["path"])
    df["annotated_path"] = df["video_id"].map(videos["annotated_path"])
    df["fps"] = df["video_id"].map(videos["fps"])
    if "best_sample" in df:
        best = samples.loc[df["best_sample"].astype(int)]
        df["best_frame"] = best["frame"].to_numpy()
        df["best_ts"] = best["ts"].to_numpy()
        df["best_crop"] = best["crop_path"].to_numpy()
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def search_by_reid_vector(index, query, top_k=20, exclude_track_pks=()):
    """Rank tracks by Re-ID similarity to one query vector (inner product = cosine)."""
    n = len(index["reid"])
    if n == 0:
        return pd.DataFrame()
    k = n  # score every stored crop; still only a few milliseconds for tens of thousands
    sims, ids = index["reid_index"].search(query.reshape(1, -1).astype(np.float32), k)
    scores = np.full(n, -1.0, dtype=np.float32)
    scores[ids[0]] = sims[0]
    df = _track_scores(index, scores)
    df = df[~df["id"].isin(exclude_track_pks)]
    df = df.sort_values("score", ascending=False).head(top_k)
    return _decorate(index, df)


def crop_query_person(detector, image_bgr):
    """Find the main person in an uploaded photo. Falls back to the whole image."""
    res = detector.predict(image_bgr, classes=[PERSON], conf=0.25, device=get_device(), verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return image_bgr, None
    xyxy = res.boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    x1, y1, x2, y2 = xyxy[int(np.argmax(areas))].astype(int)
    return image_bgr[max(0, y1):y2, max(0, x1):x2].copy(), (x1, y1, x2, y2)


def search_image(index, models, image_bgr, top_k=20):
    t0 = time.perf_counter()
    crop, box = crop_query_person(models["detector"], image_bgr)
    q = models["reid"].extract([crop])[0]
    df = search_by_reid_vector(index, q, top_k)
    return df, crop, time.perf_counter() - t0


def search_text(index, models, text, top_k=20):
    t0 = time.perf_counter()
    attrs = parse_description(text)
    q = models["clip"].encode_text(text)
    sims = index["clip"] @ q if len(index["clip"]) else np.zeros(0)
    df = _track_scores(index, sims)
    df = df.rename(columns={"score": "clip_score"})
    bonus = df.apply(lambda r: attribute_bonus(attrs, r["upper_colours"], r["lower_colours"], set(r["bags"])), axis=1)
    df["attr_bonus"] = [b for b, _ in bonus]
    df["matched_attrs"] = [", ".join(m) for _, m in bonus]
    df["score"] = df["clip_score"] + df["attr_bonus"]
    df = df.sort_values("score", ascending=False).head(top_k)
    return _decorate(index, df), attrs, time.perf_counter() - t0


def _minmax(x):
    x = np.asarray(x, dtype=np.float32)
    span = x.max() - x.min() if len(x) else 0
    return (x - x.min()) / span if span > 1e-9 else np.zeros_like(x)


def search_combined(index, models, image_bgr, text, image_weight=0.7, top_k=20):
    """Photo + description: both scores are rescaled to 0..1 over all tracks, then blended."""
    t0 = time.perf_counter()
    n_tracks = len(index["tracks"])
    img_df, crop, _ = search_image(index, models, image_bgr, top_k=n_tracks)
    txt_df, attrs, _ = search_text(index, models, text, top_k=n_tracks)
    merged = img_df[["id", "score"]].rename(columns={"score": "image_score"}).merge(
        txt_df[["id", "score", "matched_attrs"]].rename(columns={"score": "text_score"}), on="id")
    merged["score"] = image_weight * _minmax(merged["image_score"]) + (1 - image_weight) * _minmax(merged["text_score"])
    merged = merged.sort_values("score", ascending=False).head(top_k)
    df = img_df.drop(columns=["score", "rank"]).merge(merged, on="id").sort_values("score", ascending=False)
    return _decorate(index, df.drop(columns=["video", "camera"], errors="ignore")), crop, attrs, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Global identities across videos
# ---------------------------------------------------------------------------

def track_means(index):
    """Mean Re-ID vector per track (L2-normalised). Returns (track_pks, matrix)."""
    if "track_means" in index:  # cached on the index object
        return index["track_means"]
    s = index["samples"]
    pks = index["tracks"]["id"].to_numpy()
    pos = {pk: i for i, pk in enumerate(pks)}
    sums = np.zeros((len(pks), index["reid"].shape[1]), dtype=np.float32)
    np.add.at(sums, s["track_pk"].map(pos).to_numpy(dtype=np.int64),
              index["reid"][s["id"].to_numpy(dtype=np.int64)])
    sums /= np.linalg.norm(sums, axis=1, keepdims=True) + 1e-12
    index["track_means"] = (pks, sums)
    return pks, sums


def assign_global_ids(index, threshold=GLOBAL_ID_THRESHOLD):
    """Group tracks (within and across videos) that show the same person into Global IDs.

    Average-linkage agglomerative clustering: repeatedly join the two groups whose tracks
    are most similar on average (mean of all pairwise similarities), while that average is
    above `threshold`. Two groups are never joined if they contain tracks visible at the
    same time in the same video (one person cannot be in two places at once).
    """
    tracks = index["tracks"]
    pks, feats = track_means(index)
    n = len(pks)
    if n == 0:
        return {}
    vid = tracks["video_id"].to_numpy()
    f0, f1 = tracks["first_frame"].to_numpy(), tracks["last_frame"].to_numpy()

    total = (feats @ feats.T).astype(np.float64)   # sum of pairwise similarities between groups
    size = np.ones(n)
    cannot = (vid[:, None] == vid[None, :]) & ~((f1[:, None] < f0[None, :]) | (f1[None, :] < f0[:, None]))
    active = np.ones(n, dtype=bool)
    members = {i: [i] for i in range(n)}

    while True:
        avg = total / np.outer(size, size)
        avg[~active, :] = -np.inf
        avg[:, ~active] = -np.inf
        avg[cannot] = -np.inf
        np.fill_diagonal(avg, -np.inf)
        a, b = np.unravel_index(int(np.argmax(avg)), avg.shape)
        if avg[a, b] < threshold:
            break
        total[a, :] += total[b, :]
        total[:, a] += total[:, b]
        size[a] += size[b]
        cannot[a, :] |= cannot[b, :]
        cannot[:, a] |= cannot[:, b]
        active[b] = False
        members[a].extend(members.pop(b))
    groups = members

    # Number Global IDs by first appearance (video order, then time).
    order = sorted(groups.values(), key=lambda g: min((vid[i], f0[i]) for i in g))
    return {int(pks[i]): gid for gid, members in enumerate(order, start=1) for i in members}


def gallery_search(index, global_id, threshold):
    """Slides workflow: where else does this Global ID appear, and when first?

    Compares the selected identity's mean appearance with every indexed track. A track
    counts as a re-identification if it was clustered into this Global ID or scores
    above `threshold`. For each video, the earliest such track gives the first
    re-identification time.
    """
    tracks = index["tracks"]
    members = tracks[tracks["global_id"] == global_id]
    if members.empty:
        return pd.DataFrame(), pd.DataFrame()
    pks, feats = track_means(index)
    pos = {pk: i for i, pk in enumerate(pks)}
    q = feats[[pos[pk] for pk in members["id"]]].mean(0)
    q /= np.linalg.norm(q)

    # A track visible at the same time as this person in the same video is someone else.
    cand = tracks.copy()
    clash = np.zeros(len(cand), dtype=bool)
    for _, m in members.iterrows():
        clash |= ((cand["video_id"] == m["video_id"]) & (cand["first_frame"] <= m["last_frame"])
                  & (cand["last_frame"] >= m["first_frame"])).to_numpy()
    cand = cand[~clash | (cand["global_id"] == global_id).to_numpy()].copy()
    cand["score"] = [float(feats[pos[pk]] @ q) for pk in cand["id"]]
    cand = _decorate(index, cand.sort_values("score", ascending=False))
    cand["member"] = cand["global_id"] == global_id
    cand["match"] = cand["member"] | (cand["score"] >= threshold)

    first = (cand[cand["match"]].sort_values("first_ts").groupby("video_id", as_index=False).first()
             .sort_values("first_ts"))
    return cand, first
