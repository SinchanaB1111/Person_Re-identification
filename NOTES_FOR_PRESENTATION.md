# Notes for the presentation / viva

## What changed from the Review-1 slides, and why

| Slides said | Final system uses | Reason (what to say if asked) |
|---|---|---|
| YOLOv8 | **YOLO11n** (same Ultralytics library) | Newer version of the same detector family: a little more accurate at the same CPU speed. `yolov8n.pt` still works; it's a one-line config change. |
| DeepSORT | **DeepSORT-style appearance tracker** (`core/tracker.py`) | Same idea as DeepSORT: Hungarian matching on motion + appearance. It uses our own Re-ID model for appearance instead of DeepSORT's separate small CNN, so one model does both jobs. A box-overlap stage runs first so appearance is only computed when needed. We tested it against the dataset's ground truth (see Evaluation). |
| ResNet-50 Re-ID | **OSNet-AIN x1.0 trained on MSMT17** | OSNet is designed for person Re-ID. It's ~10× smaller than ResNet-50 and more accurate on cross-camera benchmarks, which matters on a CPU-only laptop. The "AIN" version adds instance normalisation so it generalises to cameras it never saw. |
| — | **CLIP (ViT-B/32)** + colour/bag attributes | Needed for the report's description-based search ("man in a blue jacket with a backpack"). |
| Video 1 → Video 2 only | Any number of videos indexed in advance; search by photo, description, both, or by picking a person | Matches the full technical report. The Video 1 → Video 2 workflow is still on the **Identity gallery** page. |
| FAISS (report) | Exact inner-product index in NumPy | Same maths as FAISS `IndexFlatIP`. FAISS crashed next to PyTorch (both ship their own OpenMP library). At this scale an exact search takes a few milliseconds. |

## Why the original code did not work
1. `reid_utils.py` built OSNet with `pretrained=True`, which loads **ImageNet classification weights, not Re-ID weights**, so the similarity scores were nearly meaningless.
2. The images were not normalised (no ImageNet mean/std), and colours were flipped BGR→RGB twice.
3. The annotated video was written with the `mp4v` codec, which browsers cannot play, so the video player stayed blank.
4. `requirements.txt` pinned CUDA-only PyTorch builds (`+cu126`), which don't install on a laptop without an NVIDIA GPU. It was also missing `streamlit` and `pandas`.
5. By default only the first 300 frames were processed, and the models were reloaded on every button press.

## How the system works (one-minute version)
1. **Offline indexing (once per video).** Each frame goes through YOLO11 (people and bags) → tracker (a local track ID per person) → every 0.5 s a crop of each person is stored → OSNet Re-ID vector, CLIP vector, upper/lower clothing colour, bags → saved in SQLite plus vector files.
2. **Global IDs.** Tracks from all videos are clustered by appearance. Two tracks that are visible at the same time in the same video can never be merged.
3. **Online search.**
   - Photo: the person is detected and cropped → OSNet vector → cosine similarity against all stored crops → best 3 crops per track averaged → ranked list.
   - Description: CLIP text vector vs stored CLIP crop vectors, plus a bonus when parsed colours or bags match.
   - Combined: a weighted blend of the two.
   - Every result has the video, camera, time, track and a clip with the person boxed in red.

## Key numbers (see README for the full table)
- Cross-camera Re-ID: **Rank-1 0.69, Rank-5 0.85, mAP 0.66** (147 people, 3 cameras).
- Photo from a camera that was never indexed: correct person in the top 5 for 4 of 5 queries.
- Description search: correct person in the top 5 for half of the descriptions.
- Search latency under 0.1 s over ~30,000 stored person crops; detection mAP@0.5 = 0.79 on COCO128.
- We first tried BoT-SORT tracking. At 2 FPS it lost close-up people completely and created 400+ IDs in 30 s, so we built the appearance-based tracker. Good point to mention: the design was driven by measurement.

## Honest limitations (examiners like hearing these)
- Scores are **similarity, not identity proof**. Many people wear dark jackets and jeans, so expect near-misses.
- The demo data is 2 frames/second, which is harder for tracking than normal 25 FPS CCTV.
- Description search is weaker than photo search: CLIP was trained on internet photos, not small CCTV crops.
- No model was trained or fine-tuned on WILDTRACK, so all numbers are cross-domain.

## Demo script (5 minutes)
1. **Videos & indexing**: show the 3 indexed cameras and play one annotated video.
2. **Search → Upload a photo**: use `eval/query_photos/C4_t100s_person378.jpg`. It comes from camera 4, which was **never indexed**, and the same man is found in cameras 1/3/5. Play a clip.
3. **Search → Describe the person**: "man in a light gray hoodie and blue jeans".
4. **Identity gallery**: pick a person in Camera-1 → first re-identification time in Camera-3 and Camera-5 → download JSON.
5. **Evaluation**: walk through the numbers and the similarity histogram.
