"""Paths and default settings shared by the indexer, the search engine and the UI."""
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
INDEX_DIR = DATA_DIR / "index"
CROPS_DIR = INDEX_DIR / "crops"
ANNOTATED_DIR = INDEX_DIR / "annotated"
CLIPS_DIR = INDEX_DIR / "clips"
DB_PATH = INDEX_DIR / "person_index.db"
REID_EMB_PATH = INDEX_DIR / "reid_embeddings.npy"
CLIP_EMB_PATH = INDEX_DIR / "clip_embeddings.npy"
EVAL_DIR = ROOT / "eval"

YOLO_WEIGHTS = MODELS_DIR / "yolo11n.pt"
REID_WEIGHTS = MODELS_DIR / "osnet_ain_x1_0_msmt17.pt"
REID_ONNX = MODELS_DIR / "osnet_ain_x1_0_msmt17.onnx"
CLIP_WEIGHTS = MODELS_DIR / "clip_vit_b32_laion2b.safetensors"
CLIP_ARCH = "ViT-B-32"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}

# COCO class ids used by the detector.
PERSON = 0
BAG_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}

# Indexing defaults.
DET_CONF = 0.30
DET_IMGSZ = 1280           # larger than 640 so small, far-away people are still found
PROCESS_FPS = 5            # frames analysed per second of video (all frames if the video is slower)
SAMPLE_INTERVAL_S = 0.5    # how often a tracked person's crop is stored and embedded
MIN_CROP_HEIGHT = 50       # crops smaller than this (pixels) are too blurry for Re-ID
MIN_TRACK_SAMPLES = 2      # tracks seen fewer times than this are dropped as noise
ANNOTATED_MAX_WIDTH = 1280

# Search defaults.
GLOBAL_ID_THRESHOLD = 0.70  # cosine similarity needed to link two tracks into one Global ID
MATCH_THRESHOLD = 0.68      # Re-ID score above which a result is flagged as a likely match


def to_rel(path) -> str:
    """Store paths relative to the project folder (with '/'), so the index still works after
    the folder is copied to another computer or operating system."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)  # outside the project folder: keep absolute


def to_abs(path) -> str:
    if path is None or path != path:  # None or NaN
        return path
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)


def get_device() -> str:
    """GPU if present (CUDA, else Apple MPS), else CPU. Override with the PS_DEVICE env var."""
    forced = os.environ.get("PS_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


for _d in (VIDEOS_DIR, INDEX_DIR, CROPS_DIR, ANNOTATED_DIR, CLIPS_DIR, EVAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)
