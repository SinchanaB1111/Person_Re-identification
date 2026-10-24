# Person Detection, Tracking and Re-Identification: Image- and Description-Based Person Search

Find a person across a collection of surveillance videos by **uploading a photo**, **typing a description**, or
**picking someone from one video**. Videos are processed once in advance. After that, every search only compares
stored person features, so results come back in milliseconds.

```
OFFLINE (once per video)
 video ─► YOLO11 person detection ─► appearance tracker ─► person crops ─┬─► OSNet-AIN Re-ID embedding (512-d)
                                                                        ├─► CLIP image embedding (512-d)
                                                                        └─► colour + bag attributes
                                    ─► SQLite metadata + vector index ─► Global IDs across all videos

ONLINE (every search)
 photo ─► detect + crop ─► Re-ID embedding ─┐
 text  ─► CLIP text embedding + attributes ─┼─► vector similarity ─► per-track ranking ─► video, camera,
 both  ─► weighted blend of the two ────────┘                                             timestamp, clip
```

## Components

| Stage | What we use | Why |
|---|---|---|
| Detection | **YOLO11n** (Ultralytics), COCO-pretrained, person + bag classes | Fast on CPU; the newer, more accurate successor to YOLOv8 with the same API |
| Tracking | **DeepSORT-style tracker** (`core/tracker.py`): IoU matching, then Re-ID appearance matching gated by walking distance | Keeps identities through fast motion, low frame rates and short occlusions |
| Re-ID | **OSNet-AIN x1.0** trained on **MSMT17**, run with ONNX Runtime | Built for cross-camera re-identification; ~5 ms per crop on CPU |
| Text search | **CLIP ViT-B/32** (OpenCLIP, LAION-2B) + colour/bag attributes | Matches free-text descriptions to person crops |
| Index | SQLite + exact inner-product vector index (NumPy) | Same results as FAISS `IndexFlatIP`; avoids FAISS/PyTorch OpenMP crashes |
| Cross-video identity | Constrained agglomerative clustering into **Global IDs** | Never merges two tracks seen at the same time in the same video |
| UI | Streamlit | Search, identity gallery, indexing and evaluation pages |

All model weights are already in `models/`, so no internet connection is needed after installation.

## Install

**Windows 10/11 (CPU is fine):** extract the ZIP, then double-click **`START_HERE_Windows.bat`**. That's all.
The first run installs Python 3.11, the Visual C++ runtime and all packages by itself (10–20 minutes, about
1 GB download). After that it starts the app in seconds and opens it in the browser. Step-by-step instructions
and troubleshooting are in `HOW_TO_RUN_ON_WINDOWS.txt`. The Python environment is kept in
`%LOCALAPPDATA%\PersonSearch\venv`; delete that folder to uninstall.

**macOS / Linux:** `./setup_mac_linux.sh` then `./run_app.sh`.

## Using the app

- **🔎 Search**
  - *Upload a photo*: the main person in the photo is detected, cropped and searched.
  - *Describe the person*: e.g. "woman in a red jacket and black trousers with a backpack".
  - *Photo + description*: a blend of both searches.
  - Each result shows the camera, track, Global ID, when the person was visible and the similarity score. **▶ Play clip** shows that moment with the person boxed in red. Results can be downloaded as JSON or CSV.
- **👥 Identity gallery**: pick a person from a reference video. The app lists every other video where that person is re-identified, with the **first re-identification time**, frame and similarity, and gives a JSON download. This was the original Video 1 → Video 2 workflow.
- **📼 Videos & indexing**: upload new videos, index them, watch the annotated output, remove videos and recompute Global IDs.
- **📊 Evaluation**: the measured metrics (see below).

Indexing from the command line works too: `python index_videos.py` indexes everything in `data/videos`.

## Included demo data

`data/videos/` contains 3 cameras from the **WILDTRACK** dataset (EPFL; Chavdarova et al., CVPR 2018). Seven
synchronised cameras film the same public square. The dataset provides 400 annotated frames per camera at
**2 frames/second** (200 s). Person IDs are consistent across cameras, which gives real ground truth for
cross-camera re-identification. `eval/prepare_wildtrack.py` rebuilds these videos from the original dataset.
`eval/query_photos/` has sample query photos for the demo. They come from cameras C2, C4 and C7, which were not indexed, so every photo search there is a true cross-camera search.

See `NOTES_FOR_PRESENTATION.md` for what changed from the original slides and a demo script.

## Evaluation

`python evaluate.py` measures, on WILDTRACK ground truth:
- detection precision and recall
- tracking MOTA, IDF1 and ID switches
- cross-camera Re-ID Rank-1, Rank-5 and mAP
- photo-search and description-search Top-1/Top-5
- Global ID grouping precision and recall
- the similarity distribution of same-person vs different-person pairs
- indexing speed and search latency

It also runs the standard COCO128 detection benchmark (person class). Results are saved to `eval/results.json`
and shown on the Evaluation page.

### Measured results (CPU only, Apple M4 Pro; WILDTRACK cameras 1, 3 and 5, 600 s of video)

| What | Metric | Result |
|---|---|---|
| Detection (COCO128, person) | Precision / Recall / mAP@0.5 / mAP@0.5:0.95 | 0.785 / 0.720 / 0.789 / 0.537 |
| Detection (WILDTRACK, IoU ≥ 0.5) | Precision / Recall | 0.712 / 0.674 |
| Tracking (WILDTRACK at 2 FPS) | MOTA / IDF1 | 0.346 / 0.439 |
| Cross-camera Re-ID (track → other cameras) | Rank-1 / Rank-5 / mAP | 0.691 / 0.846 / 0.663 |
| Photo search, single crop → other cameras | Top-1 / Top-5 | 0.301 / 0.507 |
| Photo search from never-indexed cameras | Top-1 / Top-5 | 0.40 / 0.80 (5 queries) |
| Description search | Top-1 / Top-5 / Top-10 | 0.357 / 0.500 / 0.571 (14 queries) |
| Same vs different person similarity | Mean / best threshold | 0.741 vs 0.579 / 0.68 (TPR 0.80, FPR 0.11) |
| Search latency | Photo (incl. detection) / Description | ~40–90 ms / ~35–50 ms over 29,714 stored crops |
| Indexing speed | Frames per second | 2.8 FPS on 2 FPS crowded 720p video (~30 people per frame); ~22 FPS on 10 FPS 768×576 video |

For reference, a random guess would be right 1 time in 147, so Rank-1 0.69 means most identities are matched across cameras on
the first try. Tracking numbers are low because the dataset is only 2 frames per second (people move far between
frames). On normal 25 FPS CCTV, tracking is much easier.

## Project layout

```
app.py              Streamlit UI
index_videos.py     offline indexing (CLI)
evaluate.py         metrics
core/
  detect_track.py   YOLO11 detection + tracking loop, crops, attributes, annotated video
  tracker.py        DeepSORT-style appearance tracker
  reid.py           OSNet-AIN feature extractor
  osnet_ain.py      OSNet-AIN network definition (from torchreid, MIT licence)
  text_search.py    CLIP, colour attributes, description parsing
  store.py          SQLite + vector index
  search.py         photo / text / combined search, Global IDs, cross-video gallery search
  video.py          browser-playable H.264 output, highlighted clips
  config.py         paths and thresholds
models/             yolo11n.pt, OSNet-AIN (.pt + .onnx), CLIP weights
data/videos/        input videos
data/index/         generated index (safe to delete; re-run indexing)
eval/               ground truth, evaluation queries, results
```

## Limitations

Re-ID scores are appearance similarity, **not proof of identity**. People in similar clothes can score high, and
a clothing change defeats it. Low resolution, heavy occlusion and night footage reduce accuracy. Description
search works best with clear colours and bags; small crops limit how much detail CLIP can see.
