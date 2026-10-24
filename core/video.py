"""Video helpers: browser-playable H.264 writing, highlighted clips, timestamps."""
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


def format_ts(seconds):
    seconds = max(0.0, float(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


def fit_width(frame, max_width):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = max_width / w
    return cv2.resize(frame, (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)), scale


class H264Writer:
    """Writes BGR frames to an H.264 .mp4 that every browser can play.

    Uses the ffmpeg binary bundled with imageio-ffmpeg, so it also works on Windows
    without installing ffmpeg separately (OpenCV's 'mp4v' output does not play in browsers).
    """

    def __init__(self, path, fps, size):
        w, h = size
        self.size = (w - w % 2, h - h % 2)
        self.gen = imageio_ffmpeg.write_frames(
            str(path), self.size, fps=max(fps, 1.0), codec="libx264", pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p", quality=6, macro_block_size=2,
            output_params=["-movflags", "+faststart"], ffmpeg_log_level="error")
        self.gen.send(None)

    def write(self, frame_bgr):
        if frame_bgr.shape[1] != self.size[0] or frame_bgr.shape[0] != self.size[1]:
            frame_bgr = cv2.resize(frame_bgr, self.size)
        self.gen.send(np.ascontiguousarray(frame_bgr[:, :, ::-1]))

    def close(self):
        self.gen.close()


def draw_box(frame, box, label, colour, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = max(th + 4, y1)
    cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 6, ty), colour, -1)
    cv2.putText(frame, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def render_highlight_clip(video_path, boxes_by_frame, start_frame, end_frame, out_path, label,
                          fps=None, max_width=960, hold=1):
    """Cut [start_frame, end_frame] from the source video and draw a red box on the target.

    boxes_by_frame: {frame_index: (x1, y1, x2, y2)} in original video pixels, for the analysed
    frames. A box is kept on screen for `hold` frames (the gap between analysed frames).
    """
    out_path = Path(out_path)
    cap = cv2.VideoCapture(str(video_path))
    src_fps = fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    writer = None
    idx = max(0, start_frame)
    last_box, last_seen = None, -10**9
    try:
        while idx <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in boxes_by_frame:
                last_box, last_seen = boxes_by_frame[idx], idx
            if last_box is not None and idx - last_seen < hold:
                draw_box(frame, last_box, label, (0, 0, 230), 4)
            cv2.putText(frame, format_ts(idx / src_fps), (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)
            frame, _ = fit_width(frame, max_width)
            if writer is None:
                writer = H264Writer(out_path, src_fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
            idx += 1
    finally:
        cap.release()
        if writer:
            writer.close()
    return out_path
