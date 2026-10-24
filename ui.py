"""Look and feel for the Streamlit app: design tokens, global CSS and small HTML building blocks.

The visual language follows the El-Rhema Studio design system: warm off-white canvas, near-black ink,
1px hairlines with square corners, Geist for text, Geist Mono for uppercase labels and numbers,
one red accent for the main action and green for "good".
"""
import base64
from html import escape
from pathlib import Path

import streamlit as st

INK, MUTED, TEXT2 = "#0c0c0c", "#6b6a66", "#57564f"
ACCENT, OK, WARN = "#ea2804", "#0a7d43", "#9a5c00"

CSS = """
<style>
:root{
  --bg:#fafaf9; --surface:#fff; --ink:#0c0c0c; --text2:#57564f; --muted:#6b6a66; --faint:#a9a79f;
  --line:#e4e3df; --line-soft:#efeee9; --line-strong:#d6d5d0; --frame:#161614;
  --accent:#ea2804; --accent-hover:#c92203; --ok:#0a7d43; --warn:#9a5c00; --warn-bg:#fff6e6; --warn-line:#f0c36b;
  --sans:'Geist',system-ui,sans-serif; --mono:'Geist Mono',ui-monospace,monospace;
}
html, body, .stApp {background:var(--bg); -webkit-font-smoothing:antialiased;}
[data-testid="stMainBlockContainer"]{max-width:1440px; padding:96px 24px 80px;}

/* Header + top navigation */
header[data-testid="stHeader"]{background:rgba(250,250,249,.92); backdrop-filter:blur(10px);
  -webkit-backdrop-filter:blur(10px); border-bottom:1px solid var(--line);}
[data-testid="stHeaderLogo"], [data-testid="stLogo"]{height:26px !important; width:26px !important;}
:has(> [data-testid="stHeaderLogo"]){display:flex; align-items:center; gap:10px; margin-right:18px;}
:has(> [data-testid="stHeaderLogo"])::after{content:"Person Search"; font:600 16px/1 var(--sans);
  letter-spacing:-.02em; color:var(--ink); white-space:nowrap;}
[data-testid="stTopNavLink"]{background:transparent !important; border-radius:0 !important; border-bottom:2px solid transparent;
  height:58px; padding:0 12px !important; gap:8px;}
[data-testid="stTopNavLink"] p{font:500 14px/1 var(--sans) !important; color:var(--muted);}
[data-testid="stTopNavLink"] [data-testid="stIconMaterial"]{color:var(--muted) !important; font-size:18px;}
[data-testid="stTopNavLink"]:hover p, [data-testid="stTopNavLink"]:hover [data-testid="stIconMaterial"]{color:var(--ink) !important;}
[data-testid="stTopNavLink"][aria-current="page"]{border-bottom-color:var(--ink);}
[data-testid="stTopNavLink"][aria-current="page"] p,
[data-testid="stTopNavLink"][aria-current="page"] [data-testid="stIconMaterial"]{color:var(--ink) !important;}

/* Bordered containers become flat white panels */
[data-testid="stVerticalBlockBorderWrapper"], .stVerticalBlock[class*="st-key-panel"],
div[class*="st-key-panel"]{background:var(--surface); border:1px solid var(--line) !important; border-radius:0 !important;}
div[class*="st-key-panel"]{padding:18px 20px;}
div[class*="st-key-plain"]{background:transparent; border:0 !important; padding:0;}
div[class*="st-key-card"]{background:var(--surface); border:1px solid var(--line); padding:0 0 12px; gap:10px;
  transition:border-color .15s;}
div[class*="st-key-card"]:hover{border-color:var(--ink);}
div[class*="st-key-card"] > div:not(:first-child){padding:0 12px;}
div[class*="st-key-tile"]{background:var(--surface); border:1px solid var(--line); padding:0 0 10px; gap:8px;}
div[class*="st-key-tile"] > div:not(:first-child){padding:0 10px;}
.st-key-panel_people{margin-top:-1rem; border-top:0 !important; padding:14px 14px 4px;}
div[class*="st-key-tilesel"]{border-color:var(--ink); box-shadow:inset 0 0 0 1px var(--ink);}

/* Buttons */
.stButton button, .stDownloadButton button, [data-testid="stPopover"] button, .stFormSubmitButton button{
  border-radius:0 !important; font:500 14px/1 var(--sans); padding:10px 14px; min-height:40px;
  border:1px solid var(--line-strong); background:var(--surface); color:var(--ink); box-shadow:none;}
.stButton button:hover, .stDownloadButton button:hover, [data-testid="stPopover"] button:hover{
  border-color:var(--ink); color:var(--ink); background:var(--surface);}
.stButton button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"]{
  background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600;}
.stButton button[kind="primary"]:hover, .stButton button[data-testid="stBaseButton-primary"]:hover{
  background:var(--accent-hover); border-color:var(--accent-hover); color:#fff;}
.stButton button:disabled{background:var(--line-soft); border-color:var(--line); color:var(--faint);}
.stButton button[kind="tertiary"]{border:0; background:transparent; padding:6px 0; min-height:0;}

/* Segmented control: ink outlined switch like "A · STUDIO | B · BOOTH" */
[data-testid="stButtonGroup"] > div{gap:0 !important;}
@media (max-width:640px){[data-testid="stButtonGroup"] > div{flex-wrap:wrap; row-gap:4px !important;}}
[data-testid="stButtonGroup"] button{border-radius:0 !important; border:1px solid var(--ink) !important;
  margin-left:-1px; background:var(--surface); color:var(--ink); font:500 13px/1 var(--sans); min-height:40px; padding:0 14px;}
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"],
[data-testid="stButtonGroup"] button[aria-checked="true"]{background:var(--ink) !important; color:#fff !important;}
[data-testid="stButtonGroup"] button[aria-checked="true"] *{color:#fff !important;}

/* Inputs */
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"], [data-baseweb="select"] > div,
[data-baseweb="input"]{border-radius:0 !important;}
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label{font:500 11.5px/1.2 var(--mono) !important;
  letter-spacing:.08em; text-transform:uppercase; color:var(--muted) !important;}
[data-testid="stFileUploaderDropzone"]{border-radius:0; border:1px dashed var(--line-strong); background:var(--bg);}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--ink);}
[data-testid="stSlider"] [role="slider"]{border-radius:0; box-shadow:none;}
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"]{font-family:var(--mono);}

/* Expander, alerts, dataframe, dialog, popover */
[data-testid="stExpander"] details{border-radius:0; border:1px solid var(--line); background:var(--surface);}
[data-testid="stExpander"] summary p{font:600 14px/1 var(--sans);}
[data-testid="stAlertContainer"]{border-radius:0;}
[data-testid="stDataFrame"]{border:1px solid var(--line);}
div[role="dialog"]{border-radius:0 !important;}
[data-testid="stPopoverBody"]{border-radius:0; border:1px solid var(--line); box-shadow:0 8px 24px rgba(12,12,12,.08);}
[data-testid="stProgress"] > div > div > div{border-radius:0;}
[data-testid="stCaptionContainer"]{color:var(--muted);}
[data-testid="stVideo"] video, video{background:var(--frame); border-radius:0;}
hr{border-color:var(--line) !important;}
h1,h2,h3{letter-spacing:-.03em;}

/* --- Custom blocks ------------------------------------------------------- */
.ps-eyebrow{font:500 11.5px/1 var(--mono); letter-spacing:.08em; color:var(--muted); text-transform:uppercase;}
.ps-mono{font-family:var(--mono);}
.ps-head{display:flex; flex-wrap:wrap; align-items:flex-end; justify-content:space-between; gap:16px 24px;
  padding-bottom:22px; border-bottom:1px solid var(--line); margin-bottom:4px;}
.ps-head > div{display:flex; flex-direction:column; gap:10px;}
.ps-head h1{margin:0; padding:0; font:500 clamp(32px,4vw,44px)/1 var(--sans); letter-spacing:-.035em; color:var(--ink);}
.ps-head p{margin:0; max-width:420px; font:400 13.5px/1.45 var(--sans); color:var(--muted);}

.ps-hero{background-color:var(--surface); background-image:linear-gradient(var(--line-soft) 1px,transparent 1px),
  linear-gradient(90deg,var(--line-soft) 1px,transparent 1px); background-size:40px 40px;
  border:1px solid var(--line); padding:clamp(28px,4vw,52px); display:flex; flex-direction:column; gap:22px; height:100%;}
.ps-hero h2{margin:0; padding:0; max-width:640px; font:500 clamp(34px,4.2vw,56px)/1 var(--sans); letter-spacing:-.04em;
  text-wrap:balance; color:var(--ink);}
.ps-hero p{margin:0; max-width:520px; font:400 17px/1.55 var(--sans); color:var(--text2); text-wrap:pretty;}
.ps-accent{color:var(--accent);}

.ps-panel{background:var(--surface); border:1px solid var(--line);}
.ps-panel-head{display:flex; justify-content:space-between; align-items:center; gap:12px; padding:16px 20px;
  border-bottom:1px solid var(--line);}
.ps-panel-head b{font:600 15px/1 var(--sans);}
.ps-rows .ps-row{display:grid; grid-template-columns:20px 1fr auto; gap:12px; align-items:center; padding:13px 20px;
  border-bottom:1px solid var(--line-soft);}
.ps-rows .ps-row:last-child{border-bottom:0;}
.ps-row .lbl{font:500 14px/1.3 var(--sans); color:var(--ink);}
.ps-row .sub{display:block; font:400 12px/1.3 var(--mono); color:var(--muted); margin-top:3px;}
.ps-row .val{font:400 12.5px/1.2 var(--mono); color:var(--muted); text-align:right;}
.ps-icon{width:16px; height:16px; display:block;}

.ps-kv{display:flex; justify-content:space-between; gap:12px; padding:9px 0; border-bottom:1px solid var(--line-soft);
  font:400 13px/1.3 var(--sans);}
.ps-kv:last-child{border-bottom:0;}
.ps-kv span:first-child{color:var(--text2);}
.ps-kv span:last-child{font:500 12.5px/1.3 var(--mono); color:var(--ink); text-align:right;}

.ps-stats{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); background:var(--surface);
  border:1px solid var(--line);}
.ps-stat{padding:14px 16px; display:flex; flex-direction:column; gap:10px; border-right:1px solid var(--line);
  border-bottom:1px solid var(--line); margin:0 -1px -1px 0;}
.ps-stat .k{font:500 10.5px/1.2 var(--mono); letter-spacing:.08em; color:var(--muted); text-transform:uppercase;}
.ps-stat .v{font:500 28px/1 var(--mono); letter-spacing:-.03em; color:var(--ink); font-variant-numeric:tabular-nums;}
.ps-stat .v small{font-size:.5em; color:var(--muted); letter-spacing:0; margin-left:2px;}

.ps-status{display:inline-flex; align-items:center; gap:6px; font:500 11px/1 var(--mono); letter-spacing:.06em;
  text-transform:uppercase; white-space:nowrap;}
.ps-status i{width:7px; height:7px; border-radius:999px; background:currentColor; display:inline-block;}
.ps-chip{display:inline-flex; align-items:center; gap:6px; border:1px solid var(--line-strong); background:var(--surface);
  padding:6px 9px; font:500 12px/1 var(--mono); color:var(--ink); margin:0 6px 6px 0;}
.ps-chip em{font-style:normal; color:var(--muted);}

.ps-notice{display:flex; flex-wrap:wrap; align-items:center; gap:12px; padding:14px 16px; font:500 14px/1.45 var(--sans);
  border:1px solid var(--line); background:var(--surface);}
.ps-notice .tag{font:600 11px/1 var(--mono); letter-spacing:.08em; text-transform:uppercase;}
.ps-notice.warn{background:var(--warn-bg); border-color:var(--warn-line);}
.ps-notice.warn .tag{color:var(--warn);}
.ps-notice.dark{background:var(--ink); border-color:var(--ink); color:#fff;}
.ps-notice.dark .tag{color:#ff9a85;}

/* Result card */
.ps-frame{position:relative; aspect-ratio:3/4; background:var(--frame); overflow:hidden; display:block;}
.ps-frame img{width:100%; height:100%; object-fit:contain; display:block;}
.ps-frame .tl{position:absolute; top:9px; left:10px; font:500 10.5px/1 var(--mono); letter-spacing:.08em; color:#d6d5d0;}
.ps-frame .tr{position:absolute; top:7px; right:8px;}
.ps-badge{display:inline-flex; align-items:center; gap:5px; padding:5px 7px; font:600 10px/1 var(--mono);
  letter-spacing:.08em; text-transform:uppercase;}
.ps-badge.ok{background:var(--ok); color:#fff;}
.ps-badge.weak{background:rgba(12,12,12,.7); color:#d6d5d0;}
.ps-badge.ink{background:var(--ink); color:#fff;}
.ps-card-title{display:flex; justify-content:space-between; align-items:baseline; gap:8px;}
.ps-card-title b{font:600 15px/1.2 var(--sans); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.ps-card-title span{font:500 18px/1 var(--mono); letter-spacing:-.02em;}
.ps-meter{position:relative; height:4px; background:var(--line-soft); margin:8px 0 10px;}
.ps-meter span{position:absolute; inset:0 auto 0 0; background:var(--ink);}
.ps-meter.ok span{background:var(--ok);}
.ps-meter i{position:absolute; top:-3px; bottom:-3px; width:1px; background:var(--accent);}
.ps-meta{font:400 12px/1.65 var(--mono); color:var(--muted);}
.ps-meta b{font-weight:500; color:var(--ink);}

.ps-tile-cap{display:flex; justify-content:space-between; font:500 11.5px/1 var(--mono); color:var(--ink);}
.ps-tile-cap em{font-style:normal; color:var(--muted);}

.ps-table{width:100%; border-collapse:collapse; font:400 13px/1.3 var(--sans);}
.ps-table th{text-align:left; padding:10px 20px; font:500 10.5px/1.2 var(--mono); letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); border-bottom:1px solid var(--line); white-space:nowrap;}
.ps-table td{padding:12px 20px; border-bottom:1px solid var(--line-soft); white-space:nowrap; font-family:var(--mono);
  font-size:12.5px; color:var(--text2);}
.ps-table td:first-child{font:600 14px/1.3 var(--sans); color:var(--ink);}
.ps-table tr:last-child td{border-bottom:0;}

.ps-empty{padding:28px 20px; font:400 14px/1.5 var(--sans); color:var(--muted);}
.ps-foot{margin-top:28px; padding-top:16px; border-top:1px solid var(--line); font:400 12.5px/1.5 var(--sans); color:var(--muted);}
.ps-list{margin:0; padding:0; list-style:none;}
.ps-list li{display:grid; grid-template-columns:36px 1fr; gap:10px; padding:12px 20px; border-bottom:1px solid var(--line-soft);
  font:400 14px/1.5 var(--sans); color:var(--text2);}
.ps-list li:last-child{border-bottom:0;}
.ps-list li span{font:500 12px/1.6 var(--mono); color:var(--muted);}
</style>
"""

# Lucide icons (24px grid, stroke), used inside custom HTML. Buttons use Streamlit's Material icons.
_ICONS = {
    "check": '<path d="m5 12 5 5 9-10"/>',
    "minus": '<path d="M5 12h14"/>',
    "video": '<path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5"/>'
             '<rect x="2" y="6" width="14" height="12" rx="2"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
             '<path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "scan": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/>'
            '<path d="M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="10" r="3"/><path d="M7 18c.8-2.5 2.7-4 5-4s4.2 1.5 5 4"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
           '<path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
}


def icon(name, color=INK):
    # st.html strips inline <svg>, so icons travel as data-URI images.
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
           f'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">{_ICONS[name]}</svg>')
    return f'<img class="ps-icon" alt="" src="data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}">'


def inject_css():
    st.html(CSS)


def html(markup):
    st.html(markup)


def e(text):
    return escape(str(text))


@st.cache_data(show_spinner=False, max_entries=4000)
def img_data_uri(path):
    p = Path(path)
    if not p.exists():
        return ""
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def page_head(eyebrow, title, aside=""):
    aside_html = f"<p>{aside}</p>" if aside else ""
    html(f'<div class="ps-head"><div><span class="ps-eyebrow">{e(eyebrow)}</span><h1>{e(title)}</h1></div>'
         f'{aside_html}</div>')


def status(text, color=MUTED):
    return f'<span class="ps-status" style="color:{color}"><i></i>{e(text)}</span>'


def panel_head(title, right=""):
    return f'<div class="ps-panel-head"><b>{e(title)}</b>{right}</div>'


def check_rows(rows):
    """rows: (icon, icon_color, label, sub, value_html)"""
    out = []
    for ic, col, label, sub, val in rows:
        sub_html = f'<span class="sub">{e(sub)}</span>' if sub else ""
        out.append(f'<div class="ps-row">{icon(ic, col)}<span class="lbl">{e(label)}{sub_html}</span>'
                   f'<span class="val">{val}</span></div>')
    return f'<div class="ps-rows">{"".join(out)}</div>'


def stats(items):
    """items: (label, value, unit)"""
    cells = "".join(f'<div class="ps-stat"><span class="k">{e(k)}</span><span class="v">{e(v)}'
                    f'{f"<small>{e(u)}</small>" if u else ""}</span></div>' for k, v, u in items)
    html(f'<div class="ps-stats">{cells}</div>')


def notice(text, tag="", tone=""):
    tag_html = f'<span class="tag">{e(tag)}</span>' if tag else ""
    html(f'<div class="ps-notice {tone}">{tag_html}<span style="flex:1;min-width:220px">{text}</span></div>')


def chips(pairs):
    return "".join(f'<span class="ps-chip"><em>{e(k)}</em>{e(v)}</span>' for k, v in pairs)


def frame(img_path, top_left="", badge_html=""):
    src = img_data_uri(str(img_path)) if img_path else ""
    img = f'<img src="{src}" alt="">' if src else ""
    return (f'<div class="ps-frame">{img}<span class="tl">{e(top_left)}</span>'
            f'<span class="tr">{badge_html}</span></div>')
