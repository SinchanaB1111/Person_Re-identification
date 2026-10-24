"""Build demo videos + ground truth from the WILDTRACK multi-camera dataset.

WILDTRACK (EPFL, Chavdarova et al. 2018) films one public square with 7 synchronised
cameras. Its "Image_subsets" are 400 frames per camera at 2 frames/second, and every
frame is annotated with a person ID that is the same across cameras, which gives us
real ground truth for cross-camera re-identification.

    python eval/prepare_wildtrack.py <path/to/Wildtrack_dataset> C1 C3 C5

writes data/videos/wildtrack_C1.mp4 ... and eval/wildtrack_gt.csv
"""
import csv
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import EVAL_DIR, VIDEOS_DIR  # noqa: E402
from core.video import H264Writer  # noqa: E402

FPS = 2.0
OUT_W, OUT_H = 1280, 720


def main(root, cams):
    root = Path(root)
    ann_files = sorted((root / "annotations_positions").glob("*.json"))
    scale = OUT_W / 1920
    rows = []
    for cam in cams:
        view = int(cam[1:]) - 1
        frames = sorted((root / "Image_subsets" / cam).glob("*.png"))
        out = VIDEOS_DIR / f"wildtrack_{cam}.mp4"
        writer = H264Writer(out, FPS, (OUT_W, OUT_H))
        for idx, f in enumerate(frames):
            img = cv2.imread(str(f))
            writer.write(cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA))
        writer.close()
        print(f"{out.name}: {len(frames)} frames")

        for idx, af in enumerate(ann_files[:len(frames)]):
            for person in json.loads(af.read_text()):
                v = person["views"][view]
                if v["xmax"] < 0 or v["xmin"] < 0:
                    continue
                rows.append([out.name, idx, person["personID"], round(v["xmin"] * scale, 1),
                             round(v["ymin"] * scale, 1), round(v["xmax"] * scale, 1), round(v["ymax"] * scale, 1)])

    with open(EVAL_DIR / "wildtrack_gt.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "frame", "person_id", "x1", "y1", "x2", "y2"])
        w.writerows(rows)
    print(f"ground truth: {len(rows)} boxes -> eval/wildtrack_gt.csv")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["C1", "C3", "C5"])
