"""Persistent search index: SQLite for metadata, a vector index for embeddings.

Row i of the embedding matrices belongs to samples.id == i, so a vector-search hit
maps straight back to a video, track, frame and bounding box.
"""
import json
import sqlite3
import time

import numpy as np
import pandas as pd

from .config import CLIP_EMB_PATH, DB_PATH, REID_EMB_PATH, to_abs, to_rel

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    camera TEXT,
    path TEXT,
    fps REAL,
    frames INTEGER,
    width INTEGER,
    height INTEGER,
    duration_s REAL,
    annotated_path TEXT,
    processing_s REAL,
    indexed_at TEXT
);
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY,
    video_id INTEGER,
    track_id INTEGER,
    first_frame INTEGER,
    last_frame INTEGER,
    first_ts REAL,
    last_ts REAL,
    detections INTEGER,
    samples INTEGER,
    upper_colours TEXT,
    lower_colours TEXT,
    bags TEXT,
    rep_crop TEXT,
    global_id INTEGER
);
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY,
    track_pk INTEGER,
    video_id INTEGER,
    frame INTEGER,
    ts REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    conf REAL,
    crop_path TEXT
);
CREATE TABLE IF NOT EXISTS boxes (
    video_id INTEGER,
    track_id INTEGER,
    frame INTEGER,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL
);
CREATE INDEX IF NOT EXISTS idx_boxes ON boxes(video_id, track_id);
CREATE INDEX IF NOT EXISTS idx_samples_track ON samples(track_pk);
"""


def connect():
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    return con


def _load_matrix(path):
    if path.exists():
        return np.load(path)
    return np.zeros((0, 512), dtype=np.float32)


class FlatIPIndex:
    """Exact inner-product (= cosine, since vectors are L2-normalised) nearest-neighbour search.

    Same results and API shape as faiss.IndexFlatIP. FAISS itself is not used because it
    ships its own OpenMP runtime, which crashes when loaded next to PyTorch's. A single
    matrix product over tens of thousands of 512-d vectors takes a few milliseconds.
    """

    def __init__(self, matrix):
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.ntotal = len(self.matrix)

    def search(self, queries, k):
        sims = np.atleast_2d(queries).astype(np.float32) @ self.matrix.T
        k = min(k, self.ntotal)
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k] if k < self.ntotal else np.tile(np.arange(k), (len(sims), 1))
        top = np.take_along_axis(sims, idx, axis=1)
        order = np.argsort(-top, axis=1)
        return np.take_along_axis(top, order, axis=1), np.take_along_axis(idx, order, axis=1)


def delete_video(con, name):
    """Remove a video and its rows; embeddings rows are compacted so ids stay aligned."""
    row = con.execute("SELECT id FROM videos WHERE name=?", (name,)).fetchone()
    if not row:
        return
    vid = row[0]
    reid, clip = _load_matrix(REID_EMB_PATH), _load_matrix(CLIP_EMB_PATH)
    keep = pd.read_sql("SELECT id FROM samples WHERE video_id != ? ORDER BY id", con, params=(vid,))["id"].to_numpy(dtype=np.int64)
    con.execute("DELETE FROM samples WHERE video_id=?", (vid,))
    con.execute("DELETE FROM tracks WHERE video_id=?", (vid,))
    con.execute("DELETE FROM boxes WHERE video_id=?", (vid,))
    con.execute("DELETE FROM videos WHERE id=?", (vid,))
    # Renumber the remaining samples 0..n-1 to match the compacted matrices.
    con.execute("CREATE TEMP TABLE remap AS SELECT id AS old, ROW_NUMBER() OVER (ORDER BY id) - 1 AS new FROM samples")
    con.execute("UPDATE samples SET id = -1 - (SELECT new FROM remap WHERE old = samples.id)")
    con.execute("UPDATE samples SET id = -1 - id")
    con.execute("DROP TABLE remap")
    con.commit()
    reid, clip = reid[keep], clip[keep]
    np.save(REID_EMB_PATH, reid)
    np.save(CLIP_EMB_PATH, clip)


def add_video(con, video_meta, tracks, boxes, reid_emb, clip_emb):
    """Store one processed video.

    tracks: list of dicts (see detect_track.process_video), each with a 'samples' list
    whose items line up with rows of reid_emb / clip_emb.
    """
    delete_video(con, video_meta["name"])
    cur = con.execute(
        "INSERT INTO videos (name, camera, path, fps, frames, width, height, duration_s, annotated_path,"
        " processing_s, indexed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (video_meta["name"], video_meta["camera"], to_rel(video_meta["path"]), video_meta["fps"], video_meta["frames"],
         video_meta["width"], video_meta["height"], video_meta["duration_s"], to_rel(video_meta["annotated_path"]),
         video_meta["processing_s"], time.strftime("%Y-%m-%d %H:%M:%S")))
    vid = cur.lastrowid

    reid_all, clip_all = _load_matrix(REID_EMB_PATH), _load_matrix(CLIP_EMB_PATH)
    next_id = len(reid_all)
    for t in tracks:
        cur = con.execute(
            "INSERT INTO tracks (video_id, track_id, first_frame, last_frame, first_ts, last_ts, detections,"
            " samples, upper_colours, lower_colours, bags, rep_crop) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (vid, t["track_id"], t["first_frame"], t["last_frame"], t["first_ts"], t["last_ts"], t["detections"],
             len(t["samples"]), json.dumps(t["upper"]), json.dumps(t["lower"]), json.dumps(t["bags"]),
             to_rel(t["rep_crop"])))
        tpk = cur.lastrowid
        for s in t["samples"]:
            con.execute(
                "INSERT INTO samples (id, track_pk, video_id, frame, ts, x1, y1, x2, y2, conf, crop_path)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (next_id + s["row"], tpk, vid, s["frame"], s["ts"], *s["box"], s["conf"], to_rel(s["crop"])))
    con.executemany("INSERT INTO boxes VALUES (?,?,?,?,?,?,?)", [(vid, *b) for b in boxes])
    con.commit()

    reid_all = np.concatenate([reid_all, reid_emb.astype(np.float32)])
    clip_all = np.concatenate([clip_all, clip_emb.astype(np.float32)])
    np.save(REID_EMB_PATH, reid_all)
    np.save(CLIP_EMB_PATH, clip_all)
    return vid


def load_index():
    """Everything the search engine needs, read once."""
    con = connect()
    videos = pd.read_sql("SELECT * FROM videos ORDER BY id", con)
    tracks = pd.read_sql("SELECT * FROM tracks ORDER BY id", con)
    samples = pd.read_sql("SELECT * FROM samples ORDER BY id", con)
    con.close()
    for col in ("upper_colours", "lower_colours", "bags"):
        tracks[col] = tracks[col].apply(lambda s: json.loads(s) if isinstance(s, str) else [])
    for df, col in ((videos, "path"), (videos, "annotated_path"), (tracks, "rep_crop"), (samples, "crop_path")):
        df[col] = df[col].map(to_abs)
    reid = _load_matrix(REID_EMB_PATH)
    clip = _load_matrix(CLIP_EMB_PATH)
    return dict(videos=videos, tracks=tracks, samples=samples, reid=reid, clip=clip,
                reid_index=FlatIPIndex(reid), clip_index=FlatIPIndex(clip))


def track_boxes(video_id, track_id):
    con = connect()
    rows = con.execute("SELECT frame, x1, y1, x2, y2 FROM boxes WHERE video_id=? AND track_id=?",
                       (int(video_id), int(track_id))).fetchall()
    con.close()
    return {r[0]: r[1:] for r in rows}


def set_global_ids(mapping):
    """mapping: {track_pk: global_id}"""
    con = connect()
    con.executemany("UPDATE tracks SET global_id=? WHERE id=?", [(int(g), int(t)) for t, g in mapping.items()])
    con.commit()
    con.close()
