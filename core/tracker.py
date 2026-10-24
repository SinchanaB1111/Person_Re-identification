"""Appearance-based multi-person tracker (DeepSORT-style).

Each step, detections are linked to existing tracks in two stages:

1. Overlap: predicted track box vs detection box (IoU, Hungarian matching). This is cheap
   and handles people who moved only a little since the last processed frame.
2. Appearance: for whatever is left, the Re-ID embedding of the detection is compared with
   the track's appearance memory, but only for tracks close enough that the person could
   have walked there in the elapsed time. This keeps identities through fast motion,
   low frame rates and short occlusions, where IoU-only trackers (SORT/ByteTrack/BoT-SORT)
   start a new ID.

Embeddings are only computed for detections that need stage 2, which keeps CPU cost low.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_MATCH = 0.45          # stage 1: minimum IoU with the predicted box
APPEARANCE_MATCH = 0.55   # stage 2: minimum cosine similarity with the track's appearance
MAX_SPEED_HEIGHTS = 1.3   # how far (in body heights per second) a person can plausibly move
NEW_TRACK_CONF = 0.40     # detections below this confidence never start a new track
MAX_LOST_S = 4.0          # forget a track after this long without a match


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _centre(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


class _Track:
    def __init__(self, tid, box, t, feat):
        self.id, self.box, self.t, self.feat = tid, np.asarray(box, float), t, feat
        self.vel = np.zeros(2)  # pixels per second

    def predicted(self, t):
        shift = self.vel * (t - self.t)
        return self.box + np.r_[shift, shift]

    def update(self, box, t, feat):
        box = np.asarray(box, float)
        dt = max(t - self.t, 1e-3)
        v = (_centre(box) - _centre(self.box)) / dt
        self.vel = 0.5 * self.vel + 0.5 * v
        self.box, self.t = box, t
        if feat is not None:
            self.feat = feat if self.feat is None else self.feat * 0.8 + feat * 0.2
            self.feat /= np.linalg.norm(self.feat) + 1e-12


class AppearanceTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def add_feature(self, track_id, feat):
        """Feed an extra Re-ID observation (e.g. a sampled crop) into a track's appearance memory."""
        for tr in self.tracks:
            if tr.id == track_id:
                tr.feat = feat if tr.feat is None else tr.feat * 0.8 + feat * 0.2
                tr.feat /= np.linalg.norm(tr.feat) + 1e-12
                return

    def update(self, t, boxes, confs, embed):
        """t: time in seconds. boxes: (N, 4) xyxy. confs: (N,).
        embed(list_of_indices) -> {index: 512-d vector} computes Re-ID features on demand.

        Returns {detection_index: track_id} for detections assigned to a track this step,
        and the embeddings computed along the way (so callers can reuse them).
        """
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        self.tracks = [tr for tr in self.tracks if t - tr.t <= MAX_LOST_S]
        assigned, feats = {}, {}
        free_d = list(range(len(boxes)))
        free_t = list(range(len(self.tracks)))

        # Stage 1: overlap with recently seen tracks.
        if free_d and free_t:
            pred = np.array([self.tracks[i].predicted(t) for i in free_t])
            ious = iou_matrix(pred, boxes[free_d])
            r, c = linear_sum_assignment(-ious)
            used_t, used_d = set(), set()
            for ri, ci in zip(r, c):
                if ious[ri, ci] >= IOU_MATCH:
                    ti, di = free_t[ri], free_d[ci]
                    assigned[di] = ti
                    used_t.add(ti); used_d.add(di)
            free_t = [i for i in free_t if i not in used_t]
            free_d = [i for i in free_d if i not in used_d]

        # Stage 2: appearance, gated by plausible walking distance.
        if free_d and free_t:
            feats.update(embed(free_d))
            cost = np.full((len(free_t), len(free_d)), 1e3)
            for a, ti in enumerate(free_t):
                tr = self.tracks[ti]
                if tr.feat is None:
                    continue
                pc, ph = _centre(tr.predicted(t)), tr.box[3] - tr.box[1]
                gate = ph * (0.3 + MAX_SPEED_HEIGHTS * (t - tr.t))
                for b, di in enumerate(free_d):
                    db = boxes[di]
                    dh = db[3] - db[1]
                    if not (0.6 < dh / max(ph, 1) < 1.65) or np.linalg.norm(_centre(db) - pc) > gate:
                        continue
                    sim = float(tr.feat @ feats[di])
                    if sim >= APPEARANCE_MATCH:
                        cost[a, b] = 1 - sim
            r, c = linear_sum_assignment(cost)
            used_d = set()
            for ri, ci in zip(r, c):
                if cost[ri, ci] < 1e3:
                    assigned[free_d[ci]] = free_t[ri]
                    used_d.add(free_d[ci])
            free_d = [i for i in free_d if i not in used_d]

        # New tracks for confident unmatched detections.
        new = [d for d in free_d if confs[d] >= NEW_TRACK_CONF]
        missing = [d for d in new if d not in feats]
        if missing:
            feats.update(embed(missing))
        result = {}
        for di, ti in assigned.items():
            tr = self.tracks[ti]
            tr.update(boxes[di], t, feats.get(di))
            result[di] = tr.id
        for di in new:
            tr = _Track(self.next_id, boxes[di], t, feats[di])
            self.next_id += 1
            self.tracks.append(tr)
            result[di] = tr.id
        return result, feats
