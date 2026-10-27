"""Person Detection, Tracking & Re-Identification: search UI.

    streamlit run app.py
"""
import base64
import json
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import ui
from core import store
from core.config import (CLIPS_DIR, DB_PATH, EVAL_DIR, GLOBAL_ID_THRESHOLD, MATCH_THRESHOLD, PROCESS_FPS,
                         VIDEO_EXTS, VIDEOS_DIR, get_device)
from core.search import gallery_search, search_combined, search_image, search_text
from core.video import format_ts, render_highlight_clip

ASSETS = Path(__file__).resolve().parent / "assets"

st.set_page_config(page_title="Person Search", page_icon=str(ASSETS / "logo.svg"), layout="wide")
st.logo(str(ASSETS / "logo.svg"), size="large")
ui.inject_css()

# Settings appear in the header popover of more than one page; re-assigning keeps them while that page is hidden.
for _k, _v in (("top_k", 20), ("match_thr", MATCH_THRESHOLD)):
    st.session_state[_k] = st.session_state.get(_k, _v)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading AI models (first time only) ...")
def get_models():
    from index_videos import load_models
    return load_models()


@st.cache_resource(show_spinner=False)
def _load_index(version):
    return store.load_index()


def get_index():
    version = DB_PATH.stat().st_mtime if DB_PATH.exists() else 0
    return _load_index(version)


def read_upload_image(upload):
    data = np.frombuffer(upload.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def bgr_data_uri(img):
    _, buf = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def photo_input(key, label="Photo of the person"):
    """Uploader for the query photo. Returns (image_bgr or None, name)."""
    up = st.file_uploader(label, type=["jpg", "jpeg", "png", "webp"], key=f"{key}_up")
    if up is not None:
        return read_upload_image(up), up.name
    return None, None


def short_ts(seconds):
    """00:01:02.30 -> 01:02.3 for videos under an hour."""
    t = format_ts(seconds)
    return t[3:-1] if t.startswith("00:") else t[:-1]


def index_counts(index):
    return dict(videos=len(index["videos"]), tracks=len(index["tracks"]),
                ids=int(index["tracks"]["global_id"].nunique()), crops=len(index["samples"]))


# ---------------------------------------------------------------------------
# Shared result rendering
# ---------------------------------------------------------------------------

def clip_for(row):
    """Render (once) a short clip of this track with the person boxed in red."""
    out = CLIPS_DIR / f"v{int(row['video_id'])}_t{int(row['track_id'])}.mp4"
    if not out.exists():
        fps = float(row["fps"])
        pad = int(round(2 * fps))
        start = max(0, int(row["first_frame"]) - pad)
        end = min(int(row["last_frame"]) + pad, start + int(30 * fps))
        boxes = store.track_boxes(row["video_id"], row["track_id"])
        label = f"Track {int(row['track_id'])}" + (f" | Global ID {int(row['global_id'])}" if pd.notna(row.get("global_id")) else "")
        hold = max(1, round(fps / PROCESS_FPS))
        render_highlight_clip(row["video_path"], boxes, start, end, out, label, fps=fps, hold=hold)
    return out


def gid_text(row):
    return f"{int(row['global_id'])}" if pd.notna(row.get("global_id")) else "–"


@st.dialog("Track clip", width="large")
def show_clip(row, score_label):
    left, right = st.columns([2.2, 1], gap="medium")
    with left:
        with st.spinner("Cutting clip ..."):
            clip = clip_for(row)
        if clip.exists():
            st.video(str(clip), autoplay=True, muted=True)
        else:
            ui.notice(f"Could not read the source video {ui.e(row['video_path'])}", "Missing video", "warn")
    with right:
        best = format_ts(row["best_ts"]) if pd.notna(row.get("best_ts")) else "–"
        kv = [("Camera", row["camera"]), ("Video", row["video"]), ("Track", int(row["track_id"])),
              ("Global ID", gid_text(row)), (score_label, f"{float(row['score']):.3f}"),
              ("First seen", format_ts(row["first_ts"])), ("Last seen", format_ts(row["last_ts"])),
              ("Best view", best)]
        ui.html('<span class="ps-eyebrow">Details</span>' +
                "".join(f'<div class="ps-kv"><span>{ui.e(k)}</span><span>{ui.e(v)}</span></div>' for k, v in kv))
        ui.html('<div class="ps-foot" style="margin-top:8px">The matched person is boxed in red. The clip starts '
                '2 s before they appear.</div>')


def result_card(row, key, score_label="Similarity", threshold=None, show_attrs=False):
    score = float(row["score"])
    strong = threshold is not None and score >= threshold
    badge = ("" if threshold is None else
             '<span class="ps-badge ok">Likely match</span>' if strong else '<span class="ps-badge weak">Weak</span>')
    crop = row.get("best_crop") or row.get("rep_crop")
    tick = f'<i style="left:{threshold * 100:.0f}%"></i>' if threshold is not None else ""
    attrs = ""
    if show_attrs:
        up, lo = ", ".join(row["upper_colours"]) or "?", ", ".join(row["lower_colours"]) or "?"
        attrs = f"<br>UPPER <b>{ui.e(up)}</b> · LOWER <b>{ui.e(lo)}</b>"
        if row["bags"]:
            attrs += f"<br>CARRIES <b>{ui.e(', '.join(row['bags']))}</b>"
        if row.get("matched_attrs"):
            attrs += f'<br><span style="color:{ui.OK}">MATCHED {ui.e(row["matched_attrs"])}</span>'
    with st.container(key=f"card_{key}"):
        ui.html(ui.frame(crop, f"#{int(row['rank']):02d}", badge))
        ui.html(f'<div><div class="ps-card-title"><b>{ui.e(row["camera"])}</b><span>{score:.3f}</span></div>'
                f'<div class="ps-meter{" ok" if strong else ""}"><span style="width:{min(max(score, 0), 1) * 100:.0f}%">'
                f'</span>{tick}</div>'
                f'<div class="ps-meta">TRACK <b>{int(row["track_id"])}</b> · ID <b>{gid_text(row)}</b><br>'
                f'{short_ts(row["first_ts"])} → {short_ts(row["last_ts"])}{attrs}</div></div>')
        if st.button("Play clip", key=f"play_{key}", icon=":material/play_arrow:", width="stretch"):
            show_clip(row, score_label)


def results_grid(df, prefix, cols=5, **kw):
    if df is None or df.empty:
        ui.html('<div class="ps-panel ps-empty">No results.</div>')
        return
    rows = list(df.to_dict("records"))
    for start in range(0, len(rows), cols):
        columns = st.columns(cols, gap="small")
        for col, row in zip(columns, rows[start:start + cols]):
            with col:
                result_card(row, f"{prefix}_{row['id']}", **kw)


def results_downloads(df, name, extra=None):
    cols = ["rank", "score", "camera", "video", "track_id", "global_id", "first_ts", "last_ts", "best_ts",
            "best_frame"]
    table = df[[c for c in cols if c in df.columns]].copy()
    for c in ("first_ts", "last_ts", "best_ts"):
        if c in table:
            table[c.replace("_ts", "_time")] = table[c].map(format_ts)
    payload = {"query": extra or {}, "results": json.loads(table.to_json(orient="records"))}
    with st.container(horizontal=True, key=f"plain_dl_{name}"):
        st.download_button("JSON", json.dumps(payload, indent=2), f"{name}.json", "application/json",
                           icon=":material/download:")
        st.download_button("CSV", table.to_csv(index=False), f"{name}.csv", "text/csv", icon=":material/download:")


def section_label(text, top=28):
    ui.html(f'<div class="ps-head" style="margin-top:{top}px;padding-bottom:14px"><div>'
            f'<span class="ps-eyebrow">{ui.e(text)}</span></div></div>')


def head_with_settings(eyebrow, title, aside, show_top_k=True):
    """Page header, then a toolbar row; returns the toolbar's left slot. Settings sit on the right."""
    ui.page_head(eyebrow, title, aside)
    left, right = st.columns([5, 1], vertical_alignment="center")
    with right, st.container(horizontal=True, horizontal_alignment="right", key="plain_settings"):
        with st.popover("Settings", icon=":material/tune:"):
            if show_top_k:
                st.slider("Results to show", 5, 50, step=5, key="top_k")
            st.slider("Likely-match threshold", 0.3, 0.9, step=0.01, key="match_thr",
                      help="Re-ID similarity above which a result is marked as a likely match.")
            st.caption("Scores are appearance similarity, not proof of identity. People in similar clothes can "
                       "score high, so always check the clip.")
    return left


def empty_index_notice(index):
    if index["videos"].empty:
        ui.notice("No videos are indexed yet. Open <b>Videos</b> in the top bar to add some.", "Empty index", "warn")
        return True
    return False


def index_ready_rows(index):
    c = index_counts(index)
    return ui.check_rows([
        ("check", ui.OK, "Videos indexed", "detected · tracked · embedded", f"{c['videos']}"),
        ("check", ui.OK, "Person tracks", "one per person per video", f"{c['tracks']:,}"),
        ("check", ui.OK, "Identities", "linked across videos", f"{c['ids']:,}"),
        ("check", ui.OK, "Stored views", "Re-ID + CLIP features", f"{c['crops']:,}"),
        ("cpu", ui.MUTED, "Device", "", get_device().upper()),
    ])


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

MODES = {"photo": ":material/photo_camera: Photo", "text": ":material/notes: Description",
         "both": ":material/join: Photo + description"}

HERO = {
    "photo": ("Upload a photo. <span class='ps-accent'>We'll find them</span> on every camera.",
              "The person in your photo is detected, turned into an appearance fingerprint and compared with every "
              "stored view, across all videos at once."),
    "text": ("Describe what they wore. <span class='ps-accent'>We'll find</span> who matches.",
             "Clothing colours and bags are read out of your sentence and matched directly; the full sentence is "
             "also compared with every person using CLIP."),
    "both": ("Blurry photo? <span class='ps-accent'>Add a few words</span> to sharpen it.",
             "Photo similarity and description match are each rescaled to 0–1 and blended with the weight you "
             "choose."),
}


def page_search():
    index = get_index()
    c = index_counts(index)
    toolbar = head_with_settings(f"Search · {c['videos']} videos · {c['crops']:,} stored views", "Find a person",
                                 "Videos were processed in advance, so a search only compares stored person "
                                 "features and never re-reads a video.")
    if empty_index_notice(index):
        return
    top_k, thr = st.session_state["top_k"], st.session_state["match_thr"]

    with toolbar:
        mode = st.segmented_control("Search by", list(MODES), format_func=MODES.get, default="photo",
                                    required=True, key="mode", label_visibility="collapsed")
    res_key = {"photo": "img_res", "text": "txt_res", "both": "both_res"}[mode]
    query_col, side_col = st.columns([1, 2.5], gap="medium")

    with query_col, st.container(key="panel_query"):
        if mode == "photo":
            ui.html('<span class="ps-eyebrow">Query · photo</span>')
            img, name = photo_input("img")
            st.caption("A full-body photo works best. If it shows several people, the largest one is used.")
            if st.button("Search all videos", type="primary", key="img_go", disabled=img is None,
                         icon=":material/search:", width="stretch"):
                with st.spinner("Searching ..."):
                    df, crop, secs = search_image(index, get_models(), img, top_k)
                st.session_state["img_res"] = (df, crop, secs, name)
        elif mode == "text":
            ui.html('<span class="ps-eyebrow">Query · description</span>')
            text = st.text_area("Description", placeholder="e.g. man in a blue jacket and black trousers carrying "
                                "a backpack", key="txt_in", height=110)
            st.caption("Mention clothing colours (upper and lower body), bags and headwear. Colours and bags are "
                       "matched as attributes; the rest of the sentence is matched by CLIP.")
            if st.button("Search all videos", type="primary", key="txt_go", disabled=not text.strip(),
                         icon=":material/search:", width="stretch"):
                with st.spinner("Searching ..."):
                    df, attrs, secs = search_text(index, get_models(), text, top_k)
                st.session_state["txt_res"] = (df, attrs, secs, text)
        else:
            ui.html('<span class="ps-eyebrow">Query · photo + description</span>')
            img2, _ = photo_input("both", "Photo (can be partial or blurry)")
            text2 = st.text_input("Extra details", placeholder="e.g. black backpack, white shoes", key="both_txt")
            weight = st.slider("Weight on photo", 0.0, 1.0, 0.7, 0.05, key="both_w")
            if st.button("Search all videos", type="primary", key="both_go",
                         disabled=img2 is None or not text2.strip(), icon=":material/search:", width="stretch"):
                with st.spinner("Searching ..."):
                    df, crop, attrs, secs = search_combined(index, get_models(), img2, text2, weight, top_k)
                st.session_state["both_res"] = (df, crop, secs)

    result = st.session_state.get(res_key)
    with side_col:
        if result is None:
            title, copy = HERO[mode]
            h, p = st.columns([1.3, 1], gap="medium")
            with h:
                ui.html(f'<div class="ps-hero"><span class="ps-eyebrow ps-accent">Ready · {c["videos"]} videos '
                        f'indexed</span><h2>{title}</h2><p>{copy}</p></div>')
            with p:
                ui.html(f'<div class="ps-panel">{ui.panel_head("Index", ui.status("Ready", ui.OK))}'
                        f'{index_ready_rows(index)}</div>')
            return

        if mode == "text":
            df, attrs, secs, text = result
            understood = [(k, v) for k, v in (("upper", attrs["upper"]), ("lower", attrs["lower"]),
                                               ("colours", ", ".join(attrs["any"])),
                                               ("carrying", ", ".join(attrs["bags"]))) if v]
            ui.stats([("Search time", f"{secs * 1000:.0f}", "ms"), ("People compared", f"{c['tracks']:,}", ""),
                      ("Results", f"{len(df)}", "")])
            chips = (ui.chips(understood) if understood else
                     '<span class="ps-chip"><em>attributes</em>none · CLIP text match only</span>')
            ui.html(f'<div class="ps-panel" style="margin-top:12px">{ui.panel_head("Understood from your description")}'
                    f'<div style="padding:14px 20px 8px"><p style="margin:0 0 12px;font:500 15px/1.4 var(--sans)">'
                    f'“{ui.e(text)}”</p>{chips}</div></div>')
            results_downloads(df, "description_search", {"type": "description", "text": text, "attributes": attrs})
        else:
            df, crop, secs = result[:3]
            qc, sc = st.columns([1, 4.6], gap="small")
            with qc:
                ui.html(f'<div class="ps-frame"><img src="{bgr_data_uri(crop)}" alt=""><span class="tl">QUERY</span></div>')
            with sc:
                cells = [("Search time", f"{secs * 1000:.0f}", "ms"), ("Views compared", f"{c['crops']:,}", "")]
                if mode == "photo":
                    strong = int((df["score"] >= thr).sum()) if not df.empty else 0
                    cells += [("Likely matches", f"{strong}", f"/ {len(df)}"), ("Threshold", f"{thr:.2f}", "")]
                    note = "Green results score at or above the likely-match threshold. Change it under <b>Settings</b>."
                else:
                    cells += [("Results", f"{len(df)}", "")]
                    note = "Score is a blend of photo similarity and description match, each rescaled to 0–1."
                ui.stats(cells)
                ui.html(f'<p style="margin:12px 0 0;font:400 13px/1.5 var(--sans);color:var(--muted)">{note}</p>')
                if mode == "photo":
                    results_downloads(df, "photo_search", {"type": "photo", "file": result[3]})

    section_label("Ranked by score · best view per track")
    if mode == "photo":
        results_grid(df, "img", threshold=thr)
    elif mode == "text":
        results_grid(df, "txt", score_label="Match", show_attrs=True)
    else:
        results_grid(df, "both", score_label="Combined", show_attrs=True)
    ui.html('<div class="ps-foot">Scores are appearance similarity, not proof of identity. People in similar '
            'clothes can score high — always check the clip.</div>')


def page_gallery():
    index = get_index()
    c = index_counts(index)
    toolbar = head_with_settings(f"Re-identification · {c['ids']} identities across {c['videos']} videos",
                                 "Identities", "Every track is grouped into a Global ID by appearance. Pick a "
                                 "person seen in one video to see when they are first re-identified in each other "
                                 "video.", show_top_k=False)
    if empty_index_notice(index):
        return
    tracks = index["tracks"].dropna(subset=["global_id"])
    videos = index["videos"]
    vid_names = dict(zip(videos["id"], videos["camera"]))
    thr = st.session_state["match_thr"]
    gid = st.session_state.get("gid")

    with toolbar:
        ref_cam = st.segmented_control("Reference video", list(videos["id"]), format_func=vid_names.get,
                                       default=videos["id"].iloc[0], required=True, key="ref_cam",
                                       label_visibility="collapsed")
    left, right = st.columns([1, 1.45], gap="medium")
    with left:
        ref = tracks[tracks["video_id"] == ref_cam]
        people = (ref.sort_values("samples", ascending=False).groupby("global_id", as_index=False).first()
                  .sort_values("first_ts"))
        min_samples = st.slider("Hide short tracks (fewer than N stored views)", 1, 20, 4, key="min_samples")
        people = people[people["samples"] >= min_samples]
        ui.html(f'<div class="ps-panel" style="border-bottom:0">'
                f'{ui.panel_head(f"People in {vid_names[ref_cam]}", ui.status(f"{len(people)} shown"))}</div>')
        with st.container(height=640, key="panel_people"):
            recs = people.to_dict("records")
            for start in range(0, len(recs), 3):
                cols = st.columns(3, gap="small")
                for col, r in zip(cols, recs[start:start + 3]):
                    g = int(r["global_id"])
                    selected = g == gid
                    with col, st.container(key=f"tile{'sel' if selected else ''}_{g}"):
                        ui.html(ui.frame(r["rep_crop"]))
                        ui.html(f'<div class="ps-tile-cap"><span>ID {g}</span><em>{short_ts(r["first_ts"])}</em></div>')
                        if st.button("Selected" if selected else "Select", key=f"sel_{g}", width="stretch",
                                     type="primary" if selected else "secondary",
                                     icon=":material/check:" if selected else None):
                            st.session_state["gid"] = g
                            st.rerun()

    with right:
        if gid is None or gid not in set(tracks["global_id"].astype(int)):
            ui.html('<div class="ps-hero" style="min-height:560px"><span class="ps-eyebrow ps-accent">'
                    'No person selected</span><h2>Pick a person. See every camera <span class="ps-accent">'
                    'they walked past.</span></h2><p>Choose anyone from the list on the left. You get the moment '
                    'they are first re-identified in each video, every appearance, and a clip to check each one.'
                    '</p></div>')
            return
        cand, first = gallery_search(index, gid, thr)
        t0 = cand[cand["member"]].sort_values("first_ts").iloc[0]
        found = {r["camera"]: r for _, r in first.iterrows()}
        rows = []
        for cam in videos["camera"]:
            r = found.get(cam)
            if r is None:
                rows.append(("minus", ui.MUTED, cam, "not found above the threshold", "—"))
            else:
                evidence = "same Global ID" if r["member"] else f"similarity ≥ {thr:.2f}"
                rows.append(("check", ui.OK, cam, f"{r['video']} · track {int(r['track_id'])} · {evidence}",
                             f'<b style="color:{ui.INK};font-weight:500">{format_ts(r["first_ts"])}</b><br>'
                             f'{float(r["score"]):.3f}'))
        all_found = len(found) == len(videos)
        ui.html(f'<div class="ps-panel"><div style="padding:20px 20px 18px;border-bottom:1px solid var(--line);'
                f'display:flex;flex-direction:column;gap:10px"><span class="ps-eyebrow">Global ID {gid}</span>'
                f'<span style="font:500 30px/1 var(--sans);letter-spacing:-.03em">Person {gid}</span>'
                f'<span style="font:400 14px/1.4 var(--sans);color:var(--text2)">First seen in '
                f'<b>{ui.e(t0["camera"])}</b> at <span class="ps-mono">{format_ts(t0["first_ts"])}</span> '
                f'(track {int(t0["track_id"])}).</span></div>'
                f'{ui.panel_head("First re-identified in each video", ui.status(f"{len(found)} / {len(videos)} videos", ui.OK if all_found else ui.MUTED))}'
                f'{ui.check_rows(rows)}</div>')

        summary = [{"Video": r["video"], "Camera": r["camera"], "First re-identified at": format_ts(r["first_ts"]),
                    "Seconds": round(float(r["first_ts"]), 2), "Frame": int(r["first_frame"]),
                    "Track ID": int(r["track_id"]), "Similarity": round(float(r["score"]), 3),
                    "Evidence": "same Global ID" if r["member"] else f"similarity ≥ {thr:.2f}"}
                   for _, r in first.iterrows()]
        payload = {"global_id": gid, "threshold": thr, "first_reidentification": summary,
                   "all_matches": json.loads(cand[cand["match"]][["camera", "video", "track_id", "score", "first_ts",
                                                                   "last_ts", "first_frame"]].to_json(orient="records"))}
        with st.container(horizontal=True, key="plain_gal_dl"):
            st.download_button("Re-identification results (JSON)", json.dumps(payload, indent=2),
                               f"global_id_{gid}_reid.json", "application/json", icon=":material/download:")

        matches = cand[cand["match"]].sort_values(["video_id", "first_ts"]).copy()
        matches["rank"] = np.arange(1, len(matches) + 1)
        section_label(f"All appearances · {len(matches)} tracks", top=12)
        results_grid(matches, f"gal{gid}", cols=3, threshold=thr)
        with st.expander("Closest non-matching candidates (for review)", icon=":material/rule:"):
            near = cand[~cand["match"]].head(9).copy()
            near["rank"] = np.arange(1, len(near) + 1)
            results_grid(near, f"near{gid}", cols=3, threshold=thr)


class Cancelled(Exception):
    pass


@st.cache_resource
def get_job():
    """Shared state of the background indexing job (the UI polls it)."""
    return {"thread": None, "cancel": threading.Event(), "status": "idle", "text": "", "frac": 0.0, "msg": ""}


def _run_job(job, models, items, max_s):
    from index_videos import index_one, refresh_global_ids

    def check():
        if job["cancel"].is_set():
            raise Cancelled()

    try:
        for i, (p, cam) in enumerate(items):
            t0 = time.time()

            def progress(done, total, i=i, name=p.name, t0=t0):
                check()
                job["frac"] = (i + min(done / max(total, 1), 1.0)) / len(items)
                job["text"] = f"{name} ({i + 1}/{len(items)}): frame {done}/{total} · {time.time() - t0:.0f}s"

            index_one(models, p, cam, max_s, progress)
            check()
        job["text"] = "Updating Global IDs ..."
        refresh_global_ids()
        job["status"], job["msg"] = "done", f"Processed {len(items)} video(s)."
    except Cancelled:
        job["status"], job["msg"] = "cancelled", "Processing terminated."
    except Exception as ex:  # surfaced in the UI
        job["status"], job["msg"] = "error", f"Processing failed: {ex}"


def delete_indexed_video(name):
    """Drop a video from the index and move its file into data/videos/deleted/."""
    from index_videos import refresh_global_ids
    con = store.connect()
    store.delete_video(con, name)
    con.close()
    src = VIDEOS_DIR / name
    if src.exists():
        trash = VIDEOS_DIR / "deleted"
        trash.mkdir(exist_ok=True)
        dest = trash / name
        if dest.exists():
            dest = trash / f"{dest.stem}_{int(time.time())}{dest.suffix}"
        shutil.move(str(src), str(dest))
    refresh_global_ids()


@st.fragment(run_every=1)
def job_panel():
    job = get_job()
    running = job["thread"] is not None and job["thread"].is_alive()
    if running:
        st.progress(job["frac"], text=job["text"] or "Starting ...")
        if st.button("Terminate processing", icon=":material/stop_circle:", key="terminate"):
            job["cancel"].set()
            job["text"] = "Terminating ..."
    elif job["status"] in ("done", "cancelled", "error"):
        {"done": st.success, "cancelled": st.warning, "error": st.error}[job["status"]](job["msg"])
        job["status"] = "idle"
        st.rerun()


def page_videos():
    index = get_index()
    videos = index["videos"]
    c = index_counts(index)
    job = get_job()
    running = job["thread"] is not None and job["thread"].is_alive()
    ui.page_head(f"Index · {get_device()}", "Videos",
                 "Upload surveillance footage and press Process. Detection → tracking → feature extraction runs "
                 "once per video and the results are stored; searches afterwards are fast.")

    section_label("Upload videos", top=0)
    with st.container(key="panel_add"):
        up_col, opt_col = st.columns([1.6, 1], gap="large")
        gen = st.session_state.setdefault("upload_gen", 0)
        with up_col:
            ups = st.file_uploader("Surveillance videos", type=[x[1:] for x in VIDEO_EXTS],
                                   accept_multiple_files=True, key=f"vid_up_{gen}", disabled=running)
        with opt_col:
            max_s = st.number_input("Only process the first N seconds (0 = whole video)", 0, 36000, 0, 30,
                                    disabled=running)
        label = f"Process {len(ups)} video{'s' if len(ups) != 1 else ''}" if ups else "Process"
        if st.button(label, type="primary", disabled=not ups or running, icon=":material/play_arrow:"):
            items = []
            for up in ups:
                dest = VIDEOS_DIR / Path(up.name).name
                dest.write_bytes(up.getvalue())
                items.append((dest, dest.stem))
            models = get_models()
            job.update(cancel=threading.Event(), status="running", text="Starting ...", frac=0.0, msg="")
            job["thread"] = threading.Thread(target=_run_job, args=(job, models, items, max_s), daemon=True)
            job["thread"].start()
            st.session_state["upload_gen"] = gen + 1
            st.rerun()
        job_panel()

    if videos.empty:
        return
    ui.stats([("Videos", c["videos"], ""), ("Person tracks", f"{c['tracks']:,}", ""),
              ("Identities", f"{c['ids']:,}", ""), ("Stored views", f"{c['crops']:,}", ""),
              ("Footage", f"{videos['duration_s'].sum() / 60:.1f}", "min")])
    section_label("Indexed videos")
    counts = index["tracks"].groupby("video_id").size()
    with st.container(key="panel_indexed"):
        for v in videos.itertuples(index=False):
            info, btn = st.columns([8, 1], vertical_alignment="center")
            info.markdown(f"**{v.camera}** · {v.name}  \n"
                          f"{format_ts(v.duration_s)} · {int(v.width)}×{int(v.height)} · "
                          f"{int(counts.get(v.id, 0))} person tracks · indexed {v.indexed_at}")
            if btn.button("", icon=":material/delete:", key=f"del_{v.id}", help=f"Delete {v.name}",
                          disabled=running):
                delete_indexed_video(v.name)
                st.rerun()

    watch, link = st.columns([1.6, 1], gap="medium")
    with watch, st.container(key="panel_watch"):
        ui.html('<span class="ps-eyebrow">Annotated video · boxes and track IDs</span>')
        cams = videos.set_index("id")["camera"]
        pick = st.segmented_control("Video", list(videos["id"]), format_func=cams.get,
                                    default=videos["id"].iloc[0], required=True, key="watch_pick",
                                    label_visibility="collapsed")
        path = videos.set_index("id").loc[pick, "annotated_path"]
        if Path(path).exists():
            st.video(path)
        else:
            ui.html('<div class="ps-empty">Annotated video not found on disk.</div>')
    with link, st.container(key="panel_link"):
        from index_videos import refresh_global_ids
        ui.html('<span class="ps-eyebrow">Identity linking</span>'
                '<p style="margin:10px 0 0;font:400 14px/1.5 var(--sans);color:var(--text2)">Two tracks are '
                'linked into one Global ID when their average appearance similarity is at least this value. '
                'Higher means fewer wrong merges but more split identities.</p>')
        g = st.slider("Global ID linking threshold", 0.3, 0.9, GLOBAL_ID_THRESHOLD, 0.01)
        if st.button("Recompute Global IDs", icon=":material/refresh:", width="stretch"):
            n = refresh_global_ids(g)
            st.toast(f"{n} Global IDs assigned.", icon=":material/check_circle:")
            st.rerun()


def page_eval():
    path = EVAL_DIR / "results.json"
    if not path.exists():
        ui.page_head("Evaluation", "How well it works")
        ui.notice("Run <code>python evaluate.py</code> to produce evaluation results.", "No results yet", "warn")
        return
    res = json.loads(path.read_text())
    ui.page_head(f"Evaluation · {res.get('generated_at', '')} · {res.get('device', 'cpu')}", "How well it works",
                 ui.e(res.get("dataset", "")))
    sections = list(res["sections"].items())
    for start in range(0, len(sections), 2):
        cols = st.columns(2, gap="medium")
        for col, (section, values) in zip(cols, sections[start:start + 2]):
            with col:
                if isinstance(values, dict):
                    cells = "".join(f'<div class="ps-stat"><span class="k">{ui.e(k)}</span>'
                                    f'<span class="v" style="font-size:24px">{ui.e(v)}</span></div>'
                                    for k, v in values.items())
                    body = (f'<div class="ps-stats" style="border:0;grid-template-columns:repeat(auto-fit,'
                            f'minmax(130px,1fr))">{cells}</div>')
                else:
                    body = f'<div class="ps-empty" style="color:var(--text2)">{ui.e(values)}</div>'
                ui.html(f'<div class="ps-panel" style="margin-bottom:16px">{ui.panel_head(section)}{body}</div>')

    figs = [EVAL_DIR / f for f in res.get("figures", []) if (EVAL_DIR / f).exists()]
    notes = res.get("notes") or []
    if figs or notes:
        fig_col, note_col = st.columns([1.2, 1], gap="medium")
        with fig_col:
            for p in figs:
                with st.container(key=f"panel_fig_{p.stem}"):
                    ui.html(f'<span class="ps-eyebrow">Figure · {ui.e(p.stem.replace("_", " "))}</span>')
                    st.image(str(p), width="stretch")
        with note_col:
            if notes:
                items = "".join(f'<li><span>{i:02d}</span>{ui.e(n)}</li>' for i, n in enumerate(notes, 1))
                ui.html(f'<div class="ps-panel">{ui.panel_head("Notes")}<ul class="ps-list">{items}</ul></div>')


# ---------------------------------------------------------------------------

nav = st.navigation([
    st.Page(page_search, title="Search", icon=":material/person_search:", url_path="search", default=True),
    st.Page(page_gallery, title="Identities", icon=":material/groups:", url_path="identities"),
    st.Page(page_videos, title="Videos", icon=":material/video_library:", url_path="videos"),
    st.Page(page_eval, title="Evaluation", icon=":material/analytics:", url_path="evaluation"),
], position="top")
nav.run()
