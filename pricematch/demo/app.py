"""Streamlit demo: search a product, see matched offers ranked by true
landed cost, price history, fake-discount flags and the buy-or-wait call.

  pip install -r requirements.txt
  python -m pricematch.pipeline demo --db pricematch.db

Deploys as-is on Streamlit Community Cloud (main file path:
pricematch/demo/app.py). A fresh deploy has no database, so the app seeds
the sample catalog into pricematch.db on first start.
"""
from __future__ import annotations

import argparse
import base64
import html
import re
from urllib.parse import urlsplit, urlunsplit
import sys
from pathlib import Path

try:
    import streamlit as st
except ImportError:
    sys.exit("streamlit is not installed. Run: pip install -r requirements.txt")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pricematch.agent.tools import Toolset            # noqa: E402
from pricematch.db.store import Store                 # noqa: E402


def _args() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="pricematch.db")
    args, _ = parser.parse_known_args()
    return args.db


DB_PATH = _args()

# Fresh checkout (or Streamlit Cloud deploy): no database yet, so seed one.
from pricematch.pipeline import ensure_db

ensure_db(DB_PATH)

st.set_page_config(page_title="Cipher", page_icon="🏷️", layout="wide",
                           initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------
# Theme. One amber accent (#e8a33d) on near-black in dark mode; the light
# theme keeps the same accent and layout language, just on warm white.
# --------------------------------------------------------------------------

_DARK = {
    "bg": "#0a0a0b", "panel": "#101013", "panel2": "#131316",
    "border": "#232329", "border2": "#26262c", "hover": "#34343c",
    "text": "#fafafa", "sub": "#a1a1aa", "muted": "#9a9aa3",
    "faint": "#8b8b93", "accent": "#e8a33d", "accent_text": "#e8a33d",
    "ok": "#5cbc93", "warn": "#d9a03c", "red": "#e06c6c",
    "line": "#d4d4d8", "grid": "#1c1c21",
    "input_bg": "#131316", "chip_bg": "#131316",
    "app_bg": "#0a0a0b", "sidebar_bg": "#0d0d0f",
    "band_op": "0.14",
}
_LIGHT = {
    "bg": "#fafaf8", "panel": "#ffffff", "panel2": "#f4f4f1",
    "border": "#e5e5e1", "border2": "#e8e8e3", "hover": "#c9c9c2",
    "text": "#18181b", "sub": "#52525b", "muted": "#6b6b73",
    "faint": "#6b6b73", "accent": "#e8a33d", "accent_text": "#b07208",
    "ok": "#2e8b66", "warn": "#a86e14", "red": "#c24545",
    "line": "#71717a", "grid": "#ecece7",
    "input_bg": "#ffffff", "chip_bg": "#f4f4f1",
    "app_bg": "#fafaf8", "sidebar_bg": "#f4f4f1",
    "band_op": "0.18",
}


def _css(p: dict) -> str:
    return f"""
<style>
  .block-container, div[data-testid="stMainBlockContainer"] {{
      padding-top: 3.2rem !important; max-width: 1060px;}}
  [data-testid="stHeader"] {{background: {p['app_bg']} !important;}}
  html, body, [class*="css"] {{
      font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
                   Roboto, "Helvetica Neue", sans-serif;
  }}
  footer {{visibility: hidden;}}
  #MainMenu {{visibility: hidden;}}
  .stApp {{background: {p['app_bg']} !important;}}
  section[data-testid="stSidebar"] {{background: {p['sidebar_bg']} !important;
      border-right: 1px solid {p['border']};}}
  .stMarkdown, .stMarkdown p, .stMarkdown span,
  div[data-testid="stCaptionContainer"] p {{color: {p['text']};}}
  div[data-testid="stCaptionContainer"] {{opacity: 1 !important;}}
  div[data-testid="stCaptionContainer"] p {{color: {p['muted']}; opacity: 1 !important;}}

  [data-testid="stMultiSelect"] [data-tag],
  [data-testid="stMultiSelect"] [data-tag] * {{color: #18181b !important;}}

  /* ---------- entrance motion (short + subtle) ---------- */
  @keyframes pm-rise {{from {{opacity: 0; transform: translateY(7px);}}
                      to {{opacity: 1; transform: none;}}}}
  .pm-hero, .pm-props, .pm-section, .offer-card, .verdict, .pm-chip-row {{
      animation: pm-rise 0.3s ease-out both;
  }}
  @media (prefers-reduced-motion: reduce) {{
      .pm-hero, .pm-props, .pm-section, .offer-card, .verdict, .pm-chip-row {{
          animation: none;}}
  }}

  /* ---------- hero ---------- */
  .pm-kicker {{display: flex; align-items: center; gap: 0.5rem;
              font-size: 0.75rem; font-weight: 600; letter-spacing: 0.14em;
              text-transform: uppercase; color: {p['faint']}; margin-bottom: 1.1rem;}}
  .pm-dot {{width: 7px; height: 7px; border-radius: 50%; background: {p['accent']};
           display: inline-block;}}
  .pm-title {{font-size: 3.4rem; font-weight: 750; letter-spacing: -0.038em;
             line-height: 1.0; color: {p['text']}; margin: 0;}}
  .pm-accent {{color: {p['accent_text']};}}
  .pm-tag {{color: {p['sub']}; font-size: 1.06rem; margin-top: 0.7rem;
           max-width: 46rem; line-height: 1.55;}}
  .pm-chips {{margin: 1.2rem 0 0.2rem 0; display: flex; gap: 0.5rem; flex-wrap: wrap;}}
  .chip {{display: inline-block; border: 1px solid {p['border2']};
         background: {p['chip_bg']}; color: {p['sub']};
         border-radius: 7px; padding: 0.28rem 0.7rem; font-size: 0.8rem;
         font-variant-numeric: tabular-nums;}}
  .chip b {{color: {p['text']}; font-weight: 650;}}

  /* ---------- value props ---------- */
  .pm-props {{display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem;
             margin: 1.6rem 0 0.2rem 0;}}
  .pm-prop {{border: 1px solid {p['border']}; background: {p['panel']};
            border-radius: 12px; padding: 1.0rem 1.15rem;}}
  .pm-prop .sq {{width: 8px; height: 8px; border-radius: 2px;
                background: {p['accent']}; display: inline-block; margin-bottom: 0.55rem;}}
  .pm-prop .h {{font-size: 0.95rem; font-weight: 650; color: {p['text']};
               letter-spacing: -0.01em;}}
  .pm-prop .d {{color: {p['sub']}; font-size: 0.85rem; line-height: 1.5;
               margin-top: 0.3rem;}}

  .pm-rule {{border: none; border-top: 1px solid {p['border']}; margin: 1.7rem 0 0.4rem 0;
            position: relative;}}
  .pm-rule::before {{content: ""; position: absolute; top: -1px; left: 0;
                    width: 56px; border-top: 1px solid {p['accent']};}}

  /* ---------- catalog browser ---------- */
  .pm-cat {{font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.15em;
           color: {p['muted']}; font-weight: 650; margin: 1.0rem 0 0.45rem 0;}}
  .pm-grid {{display: flex; flex-wrap: wrap; gap: 0.4rem;}}
  .pm-item {{border: 1px solid {p['border2']}; background: {p['panel2']};
            color: {p['sub']}; border-radius: 7px; padding: 0.24rem 0.62rem;
            font-size: 0.78rem;}}

  /* ---------- example chips (buttons styled as chips) ---------- */
  .pm-chip-row {{margin: 0.4rem 0 0.8rem 0;}}
  div[data-testid="stButton"] > button {{
      border-radius: 999px; border: 1px solid {p['border2']};
      background: {p['chip_bg']}; color: {p['sub']};
      white-space: normal; line-height: 1.3; height: auto;
      min-height: 2.3rem;
      font-size: 0.8rem; padding: 0.3rem 0.9rem; font-weight: 550;
      transition: border-color 120ms ease, color 120ms ease;}}
  div[data-testid="stButton"] > button p {{white-space: normal !important;}}
  div[data-testid="stLinkButton"] > a {{
      border-radius: 999px; border: 1px solid {p['border2']};
      background: {p['chip_bg']}; color: {p['sub']};
      line-height: 1.3; min-height: 2.3rem;
      font-size: 0.8rem; padding: 0.3rem 0.9rem; font-weight: 550;
      text-decoration: none; height: auto; white-space: normal;}}
  div[data-testid="stLinkButton"] > a p {{white-space: normal !important;}}
  div[data-testid="stLinkButton"] > a:hover {{
      border-color: {p['accent']}; color: {p['text']};}}
  div[data-testid="stButton"] > button:hover {{
      border-color: {p['accent']}; color: {p['text']};}}
  div[data-testid="stButton"] > button[kind="primary"] {{
      background: {p['accent']}; border-color: {p['accent']}; color: #0a0a0b;
      font-weight: 650;}}

  /* ---------- section labels ---------- */
  .pm-section {{font-size: 0.72rem; text-transform: uppercase;
               letter-spacing: 0.16em; color: {p['muted']};
               margin: 1.9rem 0 0.7rem 0; font-weight: 600;}}

  /* ---------- product header ---------- */
  .pm-prod {{display: flex; align-items: center; gap: 1.0rem;
            border: 1px solid {p['border']}; background: {p['panel']};
            border-radius: 12px; padding: 0.9rem 1.15rem; margin-top: 0.6rem;}}
  .pm-prod img {{width: 64px; height: 64px; border-radius: 10px; object-fit: cover;
                border: 1px solid {p['border2']};}}
  .pm-prod .nm {{font-size: 1.12rem; font-weight: 650; color: {p['text']};
                letter-spacing: -0.012em;}}
  .pm-prod .ct {{font-size: 0.78rem; color: {p['muted']}; margin-top: 0.15rem;}}

  /* ---------- offer cards ---------- */
  .offer-card {{display: flex; justify-content: space-between; align-items: center;
               background: {p['panel']};
               border: 1px solid {p['border']}; border-radius: 12px;
               padding: 1.0rem 1.3rem; margin-bottom: 0.55rem;
               transition: border-color 120ms ease;}}
  .offer-card:hover {{border-color: {p['hover']};}}
  .offer-card.best {{border-color: rgba(232,163,61,.45);}}
  .offer-left {{display: flex; align-items: center; gap: 1.05rem;}}
  .offer-thumb {{width: 40px; height: 40px; border-radius: 8px; object-fit: cover;
                border: 1px solid {p['border2']};}}
  .rank {{font-size: 1.05rem; font-weight: 650; color: {p['muted']}; min-width: 2.1rem;
         font-variant-numeric: tabular-nums;}}
  .best .rank {{color: {p['accent_text']};}}
  .rank small {{display: block; font-size: 0.58rem; font-weight: 650;
               letter-spacing: 0.14em; color: {p['accent_text']}; margin-top: 0.15rem;}}
  .site {{font-size: 1.05rem; font-weight: 620; color: {p['text']};
         letter-spacing: -0.01em;}}
  .breakdown {{color: {p['faint']}; font-size: 0.84rem; margin-top: 0.12rem;
              font-variant-numeric: tabular-nums;}}
  .risk {{display: inline-block; font-size: 0.72rem; border-radius: 6px;
         padding: 0.14rem 0.55rem; margin-top: 0.4rem; font-weight: 550;
         border: 1px solid transparent;}}
  .risk-ok {{color: {p['ok']}; background: rgba(92,188,147,.09);
            border-color: rgba(92,188,147,.28);}}
  .risk-warn {{color: {p['warn']}; background: rgba(217,160,60,.09);
              border-color: rgba(217,160,60,.3);}}
  .landed {{text-align: right;}}
  .landed .amt {{font-size: 1.45rem; font-weight: 700; color: {p['text']};
                letter-spacing: -0.02em; font-variant-numeric: tabular-nums;}}
  .best .landed .amt {{color: {p['accent_text']};}}
  .landed .lbl {{display: block; font-size: 0.62rem; text-transform: uppercase;
                letter-spacing: 0.13em; color: {p['muted']}; margin-top: 0.1rem;}}

  /* ---------- flag pills ---------- */
  .pill {{display: inline-block; border-radius: 7px; padding: 0.3rem 0.75rem;
         font-size: 0.8rem; font-weight: 550; margin-right: 0.5rem;
         border: 1px solid transparent; font-variant-numeric: tabular-nums;}}
  .pill-red {{color: {p['red']}; background: rgba(224,108,108,.08);
             border-color: rgba(224,108,108,.32);}}
  .pill-green {{color: {p['ok']}; background: rgba(92,188,147,.08);
               border-color: rgba(92,188,147,.3);}}
  .pill-gray {{color: {p['sub']}; background: rgba(128,128,128,.06);
              border-color: {p['border2']};}}
  .flag-detail {{color: {p['faint']}; font-size: 0.9rem; margin-top: 0.6rem;
                line-height: 1.5;}}

  /* ---------- verdict banner ---------- */
  .verdict {{border-radius: 12px; padding: 1.6rem 1.25rem; text-align: center;
            margin: 0.9rem 0 0.4rem 0;}}
  .verdict.buy {{background: rgba(92,188,147,.07);
                border: 1px solid rgba(92,188,147,.38);}}
  .verdict.wait {{background: rgba(232,163,61,.07);
                 border: 1px solid rgba(232,163,61,.4);}}
  .verdict-word {{font-size: 2.5rem; font-weight: 800; letter-spacing: 0.1em;
                 line-height: 1;}}
  .buy .verdict-word {{color: {p['ok']};}}
  .wait .verdict-word {{color: {p['accent_text']};}}
  .verdict-sub {{color: {p['sub']}; font-size: 0.95rem; margin-top: 0.55rem;}}

  /* ---------- streamlit widgets ---------- */
  div[data-testid="stTextInput"] input,
  div[data-testid="stTextArea"] textarea {{
      border-radius: 8px; background: {p['input_bg']}; color: {p['text']};
      border: 1px solid {p['border2']};}}
  div[data-testid="stSelectbox"] div:has(> input) {{
      border-radius: 8px;
      background-color: {p['input_bg']} !important;
      border-color: {p['border2']} !important;}}
  div[data-testid="stSelectbox"] input {{color: {p['text']} !important;}}
  div[data-testid="stSelectbox"] svg {{fill: {p['muted']} !important;}}
  div[data-baseweb="select"] > div {{border-radius: 8px;
      background-color: {p['input_bg']} !important;
      border-color: {p['border2']} !important;}}
  div[data-baseweb="select"] * {{color: {p['text']} !important;}}
  ul[role="listbox"], div[role="listbox"] {{
      background-color: {p['input_bg']} !important;}}
  ul[role="listbox"] li, div[role="listbox"] li, li[role="option"] {{
      color: {p['text']} !important;
      background-color: {p['input_bg']} !important;}}
  div[data-testid="stWidgetLabel"] {{color: {p['sub']} !important;
      opacity: 1 !important;}}
  div[data-testid="stWidgetLabel"] p, div[data-testid="stWidgetLabel"] label,
  div[data-testid="stWidgetLabel"] span {{
      color: {p['sub']} !important; opacity: 1 !important;}}
  [data-testid="stHeader"] button, [data-testid="stHeader"] svg {{
      color: {p['muted']} !important; fill: {p['muted']} !important;}}
  [data-testid="stFileUploaderDropzone"],
  [data-testid="stFileChip"],
  [data-testid="stFileUploader"] section,
  [data-testid="stFileUploader"] > div > div {{
      background-color: {p['input_bg']} !important;
      border-color: {p['border2']} !important;
      color: {p['text']} !important;}}
  section[data-testid="stFileUploader"] * {{color: {p['text']} !important;}}
  section[data-testid="stFileUploader"] small,
  section[data-testid="stFileUploader"] span {{color: {p['sub']} !important;}}
  [data-testid="stFileChip"] span, [data-testid="stFileChip"] small {{
      color: {p['text']} !important;}}
  button[kind="primary"], button[kind="secondary"] {{border-radius: 8px;}}
  div[data-testid="stExpander"] {{border: 1px solid {p['border']}; border-radius: 10px;
                                 background: {p['panel']};}}
  div[data-testid="stExpander"] summary:hover {{color: {p['accent_text']};}}
  div[data-testid="stExpander"] summary p {{color: {p['text']};}}
  section[data-testid="stFileUploader"] > div {{border-radius: 10px;}}
  .stTabs [data-baseweb="tab"] {{font-size: 0.92rem; font-weight: 550;
      color: {p['sub']} !important;}}
  .stTabs [aria-selected="true"] {{color: {p['text']} !important;}}
  div[data-testid="stAlert"] p {{color: {p['text']};}}
  .stProgress > div > div > div > div {{background-color: {p['accent']};}}
</style>
"""


_esc = html.escape

# --------------------------------------------------------------------------
# Catalog metadata (names + categories for the hero browser and headers)
# --------------------------------------------------------------------------
from pricematch.data import sample as _sample

_CLUSTER_META = {c["cluster_id"]: (c["name"], c["category"])
                 for c in _sample.CLUSTERS}

# Shareable links: opening the app with ?pin=<cluster_id> lands on that
# product in Search & compare (the same pin the catalog chips use). The
# value is consumed once per session so later navigation is not sticky.
_qp_pin = st.query_params.get("pin")
if _qp_pin and st.session_state.get("_consumed_pin") != _qp_pin:
    st.session_state["_consumed_pin"] = _qp_pin
    if _qp_pin in _CLUSTER_META:
        st.session_state["pin"] = _qp_pin
        st.session_state["q"] = _CLUSTER_META[_qp_pin][0]
# ?light=1 keeps the theme across a cross-tab open (a real page load starts
# a fresh session); applied once so a later toggle-off is not overridden.
if st.query_params.get("light") == "1" and not st.session_state.get("_light_qp_seen"):
    st.session_state["_light_qp_seen"] = True
    st.session_state.light = True


def _share_url(cluster_id: str) -> str:
    """App URL that opens pinned to a product, keeping the current theme."""
    u = f"?pin={cluster_id}"
    if st.session_state.get("light"):
        u += "&light=1"
    return u

_CAT_ORDER = ["electronics", "audio", "phones", "laptops", "appliances", "apparel"]
_CAT_LABEL = {"electronics": "Electronics & TVs", "audio": "Audio",
              "phones": "Phones", "laptops": "Laptops",
              "appliances": "Appliances", "apparel": "Apparel & shoes"}


@st.cache_data(show_spinner=False)
def _photo_data_uri(cluster_id: str, px: int = 160) -> str | None:
    """Small JPEG data URI of the cluster's primary catalog photo, or None."""
    from pricematch.photo import PHOTOS_DIR
    path = PHOTOS_DIR / f"{cluster_id}.jpg"
    if not path.exists():
        return None
    from PIL import Image
    import io
    img = Image.open(path).convert("RGB")
    img.thumbnail((px, px))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@st.cache_resource(show_spinner="Loading the RAG pipeline (first run with the ML extras downloads small local models)…")
def _rag():
    """One RAG pipeline per server process. Models (if installed) load once;
    the corpus is rebuilt per question so new collector data is answered on."""
    from pricematch.rag.pipeline import RagPipeline

    return RagPipeline(DB_PATH, bow=_bow_model())


@st.cache_resource(show_spinner=False)
def _bow_model():
    """Train the buy-or-wait model once per server process, not on every
    rerun (each keystroke in the search box triggers a rerun)."""
    from pricematch.buy_or_wait import BuyOrWait

    return BuyOrWait()


def _money(v: float) -> str:
    return f"₹{v:,.0f}"


@st.cache_data(show_spinner=False, ttl=60)
def _deal_of_the_day() -> dict | None:
    """Biggest genuine price drop across the catalog right now.

    A drop counts the way the alerts module defines it (below the trailing
    30-day average and a new tracked low); offers the fake-discount radar
    flags as inflated are skipped, so the deal is never a fake sale.
    """
    from pricematch.alerts import scan as _al_scan
    _ts = Toolset(DB_PATH)
    _st = Store(DB_PATH)
    try:
        offs = _st.offers()
        hists = {}
        for oid in offs["offer_id"]:
            h = _st.history(oid)
            hists[oid] = [{"ts": str(r.ts), "price": float(r.price)}
                          for r in h.itertuples()]
        for a in _al_scan(hists, current_prices=dict(zip(offs["offer_id"], offs["price"]))):
            chk = _ts.price_check(a["offer_id"])
            if "inflated_before_sale" in chk["flags"]:
                continue
            row = offs[offs["offer_id"] == a["offer_id"]].iloc[0]
            return {"cluster_id": row["cluster_id"], "site": row["site"],
                    "latest": a["latest"], "avg": a["trailing_avg"],
                    "pct": a["pct_below_avg"], "new_low": a["new_low"]}
        return None
    finally:
        _st.close()
        _ts.close()


# --------------------------------------------------------------------------
# Theme state + stylesheet
# --------------------------------------------------------------------------
if "light" not in st.session_state:
    st.session_state.light = False

with st.sidebar:
    st.markdown("**Cipher** · Sahi daam, every time.")
    st.toggle("Light mode", key="light")
    st.caption("Same amber, brighter room.")

_P = _LIGHT if st.session_state.light else _DARK
st.markdown(_css(_P), unsafe_allow_html=True)

# Light-mode contrast repairs. The global Streamlit base theme is dark
# (.streamlit/config.toml), so widget chrome that the main stylesheet does
# not restyle keeps dark-theme colors and disappears on the light page.
# This block loads ONLY in light mode; dark mode is untouched.
if st.session_state.light:
    st.markdown(f"""
<style>
  /* sidebar collapse/expand arrow: amber chip + dark icon reads on the
     light page, the light sidebar, and a still-dark sidebar alike */
  div[data-testid="stSidebarCollapseButton"] button,
  button[data-testid="stExpandSidebarButton"],
  [data-testid="stSidebarCollapsedControl"] button {{
      background: {_LIGHT['accent']} !important;
      border-radius: 8px !important;
      color: #18181b !important;}}
  div[data-testid="stSidebarCollapseButton"] button svg,
  button[data-testid="stExpandSidebarButton"] svg,
  [data-testid="stSidebarCollapsedControl"] button svg,
  div[data-testid="stSidebarCollapseButton"] button span,
  button[data-testid="stExpandSidebarButton"] span,
  [data-testid="stSidebarCollapsedControl"] button span {{
      color: #18181b !important; fill: #18181b !important;}}

  /* Link buttons share dark-base widget label styles across agent answers,
     comparison cards, deal and catalog. Paint both anchor and nested label
     explicitly, including visited/hover/focus, rather than inherit white. */
  [data-testid="stLinkButton"] a,
  [data-testid="stLinkButton"] a:visited,
  a[data-testid^="stBaseLinkButton"] {{
      background: {_LIGHT['chip_bg']} !important;
      color: {_LIGHT['text']} !important;
      border-color: {_LIGHT['border2']} !important;}}
  [data-testid="stLinkButton"] a *,
  a[data-testid^="stBaseLinkButton"] * {{
      color: {_LIGHT['text']} !important;
      -webkit-text-fill-color: {_LIGHT['text']} !important;
      opacity: 1 !important;}}
  [data-testid="stLinkButton"] a:hover,
  [data-testid="stLinkButton"] a:focus-visible,
  a[data-testid^="stBaseLinkButton"]:hover,
  a[data-testid^="stBaseLinkButton"]:focus-visible {{
      background: {_LIGHT['chip_bg']} !important;
      color: {_LIGHT['text']} !important;
      border-color: {_LIGHT['accent']} !important;}}

  /* widget labels: the testid sits on a <label>, not a div, in newer
     Streamlit, so the main stylesheet's div-qualified rule never matches */
  [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
  [data-testid="stWidgetLabel"] span, [data-testid="stWidgetLabel"] label,
  [data-testid="stWidgetLabel"] div {{
      color: {_LIGHT['sub']} !important; opacity: 1 !important;}}

  /* inactive tabs are div[role=tab], not button[data-baseweb=tab] */
  [data-testid="stTab"], [data-testid="stTab"] p, [data-testid="stTab"] span,
  .stTabs [role="tab"], .stTabs [role="tab"] p, .stTabs [role="tab"] span {{
      color: {_LIGHT['sub']} !important;}}
  [data-testid="stTab"][aria-selected="true"],
  [data-testid="stTab"][aria-selected="true"] p,
  [data-testid="stTab"][aria-selected="true"] span,
  .stTabs [role="tab"][aria-selected="true"],
  .stTabs [role="tab"][aria-selected="true"] p,
  .stTabs [role="tab"][aria-selected="true"] span {{
      color: {_LIGHT['text']} !important;}}

  /* catalog expander: the dark base paints the open header near-black and
     its label near-white on the light page */
  [data-testid="stExpander"] summary,
  [data-testid="stExpander"] details summary,
  [data-testid="stExpander"] details[open] summary {{
      background: {_LIGHT['panel']} !important;
      color: {_LIGHT['text']} !important;}}
  [data-testid="stExpander"] summary p,
  [data-testid="stExpander"] summary span {{
      color: {_LIGHT['text']} !important;}}
  [data-testid="stExpander"] summary svg {{
      fill: {_LIGHT['muted']} !important; color: {_LIGHT['muted']} !important;}}

  /* photo uploader: the Browse/Upload secondary button and the dropzone
     instructions keep dark-theme colors on the light dropzone */
  [data-testid="stFileUploaderDropzone"] button {{
      background: {_LIGHT['panel2']} !important;
      color: {_LIGHT['text']} !important;
      border: 1px solid {_LIGHT['border2']} !important;}}
  [data-testid="stFileUploaderDropzone"] button * {{
      color: {_LIGHT['text']} !important;}}
  [data-testid="stFileUploaderDropzoneInstructions"],
  [data-testid="stFileUploaderDropzoneInstructions"] * {{
      color: {_LIGHT['sub']} !important;}}

  /* captions (st.caption) sit at ~3:1 on the light page - bump to sub */
  div[data-testid="stCaptionContainer"] p,
  section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
      color: {_LIGHT['sub']} !important;}}

  /* alert boxes (st.warning/info/success/error) keep dark surfaces in
     light mode: dark bar, unreadable text */
  [data-testid="stAlert"],
  [data-testid="stAlertContainer"] {{
      background: {_LIGHT['panel']} !important;
      border: 1px solid {_LIGHT['border2']} !important;
      border-radius: 10px !important;
      color: {_LIGHT['text']} !important;}}
  [data-testid="stAlert"] p, [data-testid="stAlert"] span,
  [data-testid="stAlert"] div, [data-testid="stAlert"] a,
  [data-testid="stAlertContainer"] * {{
      color: {_LIGHT['text']} !important;
      -webkit-text-fill-color: {_LIGHT['text']} !important;
      opacity: 1 !important;}}

  /* code blocks (the share-link box) keep a dark surface on the light page */
  [data-testid="stCode"], [data-testid="stCode"] pre,
  .stCode, .stCode pre {{
      background: {_LIGHT['panel2']} !important;
      border: 1px solid {_LIGHT['border2']} !important;}}
  [data-testid="stCode"] code, .stCode code,
  [data-testid="stCode"] code span, .stCode code span {{
      color: {_LIGHT['text']} !important;}}

  [data-testid="stTextInputRootElement"],
  [data-testid="stTextAreaRootElement"],
  [data-testid="stNumberInputContainer"],
  [data-testid="stNumberInputContainer"] input,
  [data-testid="stNumberInputContainer"] button {{
      background: {_LIGHT['input_bg']} !important;
      color: {_LIGHT['text']} !important;}}
  [data-testid="stProgress"] p {{color: {_LIGHT['sub']} !important;}}
  [data-testid="stProgressBarTrack"] {{background: {_LIGHT['border']} !important;}}
  [data-testid="stFileChip"] *,
  [data-testid="stMultiSelect"] [data-baseweb="tag"] * {{
      color: {_LIGHT['text']} !important;}}

  .best .rank small {{color: #995f00 !important;}}
  [data-testid="stElementToolbarButtonContainer"] {{
      background: {_LIGHT['panel2']} !important;}}
  [data-testid="stElementToolbarButtonContainer"] button,
  [data-testid="stElementToolbarButtonContainer"] svg {{color: {_LIGHT['text']} !important;}}

  /* The React-Aria picker input, placeholder and arrow have independent
     dark-base colors; painting only the outer group leaves them invisible. */
  [data-testid="stMultiSelect"] [role="group"],
  [data-testid="stMultiSelect"] input,
  [data-testid="stMultiSelect"] button {{
      background: {_LIGHT['input_bg']} !important;
      color: {_LIGHT['text']} !important;}}
  [data-testid="stMultiSelect"] input,
  [data-testid="stSelectbox"] input {{
      -webkit-text-fill-color: {_LIGHT['text']} !important;}}
  [data-testid="stMultiSelect"] input::placeholder,
  [data-testid="stSelectbox"] input::placeholder,
  [data-testid="stTextInput"] input::placeholder,
  [data-testid="stTextArea"] textarea::placeholder {{
      color: {_LIGHT['sub']} !important;
      -webkit-text-fill-color: {_LIGHT['sub']} !important;
      opacity: 1 !important;}}
  [data-testid="stMultiSelect"] button svg {{
      color: {_LIGHT['sub']} !important; fill: {_LIGHT['sub']} !important;}}
  [data-testid="stButton"] button *,
  [data-testid="stNumberInputContainer"] button * {{color: inherit !important;}}

  /* multiselect control box is near-black and dropdown options (multiselect
     and selectbox alike) are white-on-white in light mode */
  [data-testid="stMultiSelect"] > div > div:first-child {{
      background: {_LIGHT['input_bg']} !important;}}
  [role="listbox"], div[data-baseweb="popover"],
  [data-testid="stMultiSelectDropdown"],
  [data-testid="stSelectboxVirtualDropdown"] {{
      background: {_LIGHT['panel']} !important;}}
  [role="listbox"] [role="option"], [role="listbox"] [role="option"] *,
  [role="listbox"] li, [role="listbox"] li *,
  div[data-baseweb="popover"] [role="option"],
  div[data-baseweb="popover"] [role="option"] *,
  div[data-baseweb="popover"] li,
  div[data-baseweb="popover"] li * {{
      color: {_LIGHT['text']} !important;}}
</style>
""", unsafe_allow_html=True)

store = Store(DB_PATH)
_offers = store.offers()
_n_products = int(_offers["cluster_id"].nunique())
_n_sites = int(_offers["site"].nunique())


def _seed(widget_key: str, value: str) -> None:
    """Fill a widget from a chip click; a fresh search clears any pin."""
    st.session_state[widget_key] = value
    st.session_state.pop("pin", None)


def _pin_product(cluster_id: str) -> None:
    """Catalog-browser chip: open this exact product in Search & compare.

    Goes straight to the cluster id instead of relying on text search -
    a name query does not always rank the product first, and landing on
    the wrong product is worse than not jumping at all.
    """
    st.session_state["pin"] = cluster_id
    st.session_state["q"] = _CLUSTER_META[cluster_id][0]

# patch sanity: Streamlit re-runs this script on file changes but keeps
# imported modules cached, so after replacing files an old process can serve
# an old catalog against a new database (or skip a needed reseed). Comparing
# the on-disk catalog to the in-memory one catches that and says what to do.
if _n_products != len(_sample.CLUSTERS):
    st.warning(
        f"Catalog mismatch: the database holds {_n_products} products but this "
        f"running app knows {len(_sample.CLUSTERS)}. You are probably looking at "
        "an app instance that was started before the latest patch. Fully stop "
        "EVERY streamlit terminal (Ctrl+C or close the windows), start the app "
        "again, and hard-refresh the browser (Ctrl+Shift+R).")

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown(
    '<div class="pm-hero">'
    '<div class="pm-kicker"><span class="pm-dot"></span>Live demo · sample catalog</div>'
    '<div class="pm-title">C<span class="pm-accent">ipher</span></div>'
    '<div class="pm-tag">Sahi daam, every time.</div>'
    f'<div class="pm-chips"><span class="chip"><b>{_n_products}</b> products tracked</span>'
    f'<span class="chip"><b>{_n_sites}</b> stores</span>'
    '<span class="chip"><b>60-day</b> price history</span>'
    '<span class="chip"><b>3</b> ways to search</span></div>'
    '</div>'
    '<div class="pm-props">'
    '<div class="pm-prop"><span class="sq"></span>'
    '<div class="h">True landed cost</div>'
    '<div class="d">Price + shipping − coupon, ranked across stores so the '
    'cheapest sticker is not mistaken for the cheapest order.</div></div>'
    '<div class="pm-prop"><span class="sq"></span>'
    '<div class="h">Fake-discount radar</div>'
    '<div class="d">60 days of price history expose inflated "sale" prices, '
    'and a model calls buy now or wait.</div></div>'
    '<div class="pm-prop"><span class="sq"></span>'
    '<div class="h">Photo search</div>'
    '<div class="d">Snap a product and CLIP matches it offline. Unsure means '
    'unsure - never a confident price for the wrong product.</div></div>'
    '</div>',
    unsafe_allow_html=True,
)

_deal = _deal_of_the_day()
if _deal:
    _dnm = _CLUSTER_META.get(_deal["cluster_id"], (_deal["cluster_id"], ""))[0]
    st.markdown(
        '<div class="pm-section">Deal of the day · biggest tracked drop the '
        'fake-discount radar does not flag</div>'
        f'<div class="offer-card best">'
        f'<div class="offer-left">'
        f'<div class="rank">★<small>DEAL</small></div>'
        f"<div><div class=\"site\">{_esc(_dnm)}</div>"
        f'<div class="breakdown">{_esc(_deal["site"])} · {_money(_deal["latest"])} · '
        f'{_deal["pct"]}% below its 30-day average of {_money(_deal["avg"])}'
        f'{" · new tracked low" if _deal["new_low"] else ""}</div>'
        f'<span class="risk risk-ok">passes the fake-discount radar</span>'
        f"</div></div>"
        f'<div class="landed"><span class="amt">{_money(_deal["latest"])}</span>'
        f'<span class="lbl">today at {_esc(_deal["site"])}</span></div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.link_button(f"Open {_dnm} in Search & compare", _share_url(_deal["cluster_id"]),
                   key="deal-open")

with st.expander(f"Browse the full catalog · {_n_products} products"):
    _flt = st.text_input("Filter the catalog", key="catfilter",
                         placeholder="Type a name, brand or category", live=True)
    st.caption("Click a product to open it in Search & compare.")
    _by_cat: dict[str, list[tuple[str, str]]] = {}
    _f = _flt.strip().lower()
    for _cid, (_nm, _cat) in sorted(_CLUSTER_META.items(),
                                    key=lambda kv: kv[1][1]):
        if _f and _f not in f"{_nm} {_CAT_LABEL.get(_cat, _cat)}".lower():
            continue
        _by_cat.setdefault(_cat, []).append((_cid, _nm))
    if not _by_cat:
        st.caption(f"No tracked products match '{_flt.strip()}'.")
    for _cat in _CAT_ORDER + sorted(set(_by_cat) - set(_CAT_ORDER)):
        if _cat not in _by_cat:
            continue
        _label = _CAT_LABEL.get(_cat, _cat.title())
        _items = sorted(_by_cat[_cat], key=lambda t: t[1])
        st.markdown(f'<div class="pm-cat">{_esc(_label)} · {len(_items)}</div>',
                    unsafe_allow_html=True)
        for _r in range(0, len(_items), 4):
            _cols = st.columns(4)
            for _col, (_cid, _nm) in zip(_cols, _items[_r:_r + 4]):
                _col.link_button(_nm, _share_url(_cid), key=f"cat-{_cid}",
                                 use_container_width=True)

st.markdown('<hr class="pm-rule">', unsafe_allow_html=True)

tab_search, tab_agent, tab_photo, tab_compare = st.tabs(
    ["Search & compare", "Ask the agent", "Photo search", "Compare"])

_FLAG_LABELS = {
    "inflated_before_sale": ("Fake-discount risk", "pill-red"),
    "genuine_drop": ("Genuine price drop", "pill-green"),
    "insufficient_history": ("Limited history", "pill-gray"),
}

# Verified to put the named product first in search_products().
_SEARCH_EXAMPLES = [
    ("Airdopes", "boAt Airdopes 141"),
    ("iPhone 15", "iPhone 15"),
    ("TCL 4K TV", "TCL P635"),
    ("Watch", "smartwatch"),
    ("Airfryer", "Philips airfryer HD9200"),
    ("Fan", "ceiling fan"),
    ("Levi's 511", "Levis 511"),
    ("Kurti", "kurti"),
]

_AGENT_EXAMPLES = [
    ("Airdopes under ₹1500?", "boAt airdopes under 1500"),
    ("4K TV under ₹25,000?", "43 inch 4K TV under 25000"),
    ("Are these reviews genuine?", "Are the reviews for Noise ColorFit Pro 5 genuine?"),
    ("Watch under ₹3500?", "which smartwatch under 3500 is the best buy"),
]


with tab_search:
    st.session_state.setdefault("q", "boAt Airdopes 141")
    query = st.text_input("Search the tracked catalog", key="q")
    st.markdown('<div class="pm-chip-row">', unsafe_allow_html=True)
    _cols = st.columns(len(_SEARCH_EXAMPLES))
    for _col, (_label, _q) in zip(_cols, _SEARCH_EXAMPLES):
        _col.button(_label, key=f"chip-{_label}",
                    on_click=_seed, args=("q", _q))
    st.markdown('</div>', unsafe_allow_html=True)
    toolset = Toolset(DB_PATH)
    _pin = st.session_state.get("pin")
    if _pin and query != _CLUSTER_META.get(_pin, ("", ""))[0]:
        st.session_state.pop("pin", None)
        _pin = None
    if query:
        hits = toolset.search_products(query)
        if not hits:
            st.warning("Nothing tracked matches that yet.")
        else:
            titles = {}
            for h in hits:
                titles.setdefault(h["cluster_id"], h["title"])
            if _pin and _pin in _CLUSTER_META:
                # Catalog-browser chip: force the exact cluster to the top of
                # the dropdown so the user always lands on the product they
                # clicked, even when text search ranks something else first.
                titles.pop(_pin, None)
                titles = {_pin: _CLUSTER_META[_pin][0], **titles}
                st.caption("Opened from the catalog browser · "
                           f"{_CLUSTER_META[_pin][0]}")
            clusters = list(titles)  # search relevance order, best hit first
            cluster_id = st.selectbox(
                "Matched product", clusters,
                format_func=lambda c: titles[c],
            )

            _name, _cat = _CLUSTER_META.get(
                cluster_id, (cluster_id.replace("-", " "), ""))
            _uri = _photo_data_uri(cluster_id)
            _img = (f'<img src="{_uri}" alt="">' if _uri else "")
            st.markdown(
                f'<div class="pm-prod">{_img}'
                f'<div><div class="nm">{_esc(_name)}</div>'
                f'<div class="ct">{_esc(_CAT_LABEL.get(_cat, _cat.title()))}'
                f' · {len(toolset.cluster_offers(cluster_id))} store offers</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            with st.expander("Share this product"):
                st.caption("Opens this exact product. Localhost links work only on "
                           "this computer while Cipher is running. To share with "
                           "someone else, deploy the app at a reachable address first.")
                _location = urlsplit(st.context.url)
                _base_url = urlunsplit((_location.scheme, _location.netloc,
                                       _location.path, "", ""))
                st.code(_base_url + _share_url(cluster_id), language=None)

            ranked = toolset.landed_costs(cluster_id)
            best = ranked[0]["landed_cost"] if ranked else None
            st.markdown(
                f'<div class="pm-section">True landed cost · cheapest first · '
                f'best {_money(best) if best is not None else "—"}</div>',
                unsafe_allow_html=True,
            )
            _thumb = (f'<img class="offer-thumb" src="{_uri}" alt="">'
                      if _uri else "")
            for i, row in enumerate(ranked, start=1):
                ship = "free shipping" if not row["shipping"] else f"+{_money(row['shipping'])} shipping"
                coupon = f" − {_money(row['coupon_value'])} coupon" if row["coupon_value"] else ""
                risk_ok = row["risk"] == "free returns"
                best_tag = "<small>BEST PRICE</small>" if i == 1 else ""
                st.markdown(
                    f'<div class="offer-card{" best" if i == 1 else ""}" '
                    f'style="animation-delay:{i * 45}ms">'
                    f'<div class="offer-left">'
                    f'<div class="rank">#{i}{best_tag}</div>{_thumb}'
                    f"<div><div class=\"site\">{_esc(row['site'])}</div>"
                    f'<div class="breakdown">{_money(row["price"])} · {ship}{coupon}</div>'
                    f'<span class="risk {"risk-ok" if risk_ok else "risk-warn"}">{_esc(row["risk"])}</span>'
                    f"</div></div>"
                    f'<div class="landed"><span class="amt">{_money(row["landed_cost"])}</span>'
                    f'<span class="lbl">landed cost</span></div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            offers = toolset.cluster_offers(cluster_id)
            pick = st.selectbox("Offer to inspect", [o["site"] for o in offers])
            offer = next(o for o in offers if o["site"] == pick)
            history = store.history(offer["offer_id"])

            st.markdown('<div class="pm-section">Price history · 14-day forecast</div>',
                        unsafe_allow_html=True)
            import altair as alt  # bundled with streamlit; safe offline
            import pandas as pd
            from pricematch.forecast import forecast as _forecast
            _pts = [{"ts": r.ts.date().isoformat(), "price": float(r.price)}
                    for r in history.itertuples()]
            _hdf = pd.DataFrame(_pts)
            _base = (
                alt.Chart(_hdf)
                .mark_line(color=_P["line"], strokeWidth=2)
                .encode(x=alt.X("ts:T", title=None,
                                axis=alt.Axis(format="%b %d", tickCount=6)),
                        y=alt.Y("price:Q", title="₹",
                                scale=alt.Scale(zero=False)))
            )
            _dots = (
                alt.Chart(_hdf)
                .mark_point(color=_P["line"], size=14, opacity=0.55)
                .encode(x="ts:T", y="price:Q")
            )
            _fc = _forecast(_pts)
            if _fc["points"]:
                _fdf = pd.DataFrame(_fc["points"])
                _band = alt.Chart(_fdf).mark_area(
                    color="#e8a33d", opacity=float(_P["band_op"])).encode(
                    x="ts:T", y="lower:Q", y2="upper:Q")
                _fline = alt.Chart(_fdf).mark_line(
                    color="#e8a33d", strokeWidth=2, strokeDash=[5, 4]).encode(
                    x="ts:T", y="predicted:Q")
                _chart = (_base + _dots + _band + _fline).properties(
                    height=400, background=_P["bg"]).configure_axis(
                    labelColor=_P["sub"], titleColor=_P["sub"],
                    gridColor=_P["grid"], domainColor=_P["border"],
                    tickColor=_P["border"], labelFontSize=11,
                ).configure_view(strokeWidth=0)
                st.altair_chart(_chart, use_container_width=True)
                st.caption(
                    f"Dashed: 14-day projection of trend + weekly seasonality "
                    f"(weekly drift {_money(_fc['trend_per_week']) if _fc['trend_per_week'] is not None else '—'}). "
                    "Shaded band ≈ 80% interval, widening with horizon. A projection, not a promise.")
            else:
                st.altair_chart((_base + _dots).properties(
                    height=400, background=_P["bg"]).configure_axis(
                    labelColor=_P["sub"], titleColor=_P["sub"],
                    gridColor=_P["grid"], domainColor=_P["border"],
                    tickColor=_P["border"], labelFontSize=11,
                ).configure_view(strokeWidth=0), use_container_width=True)

            check = toolset.price_check(offer["offer_id"])
            pills = []
            for flag in check["flags"]:
                label, cls = _FLAG_LABELS.get(flag, (flag.replace("_", " "), "pill-gray"))
                pills.append(f'<span class="pill {cls}">{_esc(label)}</span>')
            st.markdown(
                '<div class="pm-section">Price check</div>'
                + "".join(pills)
                + f'<div class="flag-detail">{_esc(check["detail"])}</div>',
                unsafe_allow_html=True,
            )

            from pricematch.review_intel import summarize as _review_sum
            from pricematch.review_intel import trust as _review_trust
            _rt = _review_trust(cluster_id)
            if _rt["score"] is not None:
                _rt_cls = {"high": "pill-green", "medium": "pill-gray",
                           "low": "pill-red"}[_rt["label"]]
                _rt_detail = "; ".join(f["detail"] for f in _rt["flags"])
                st.markdown(
                    '<div class="pm-section">Review trust</div>'
                    f'<span class="pill {_rt_cls}">'
                    f'{_esc(_rt["label"].title())} · {_rt["score"]}/100 · '
                    f'{_rt["review_count"]} reviews, avg {_rt["avg_rating"]}★</span>'
                    + (f'<div class="flag-detail">Manipulation signals: {_esc(_rt_detail)}</div>' if _rt["flags"] else ""),
                    unsafe_allow_html=True,
                )
                with st.expander("What reviewers say · extractive pros & cons"):
                    _rs = _review_sum(cluster_id)
                    _c1, _c2 = st.columns(2)
                    _c1.markdown("**Pros**")
                    for p in _rs["pros"]:
                        _c1.markdown(f"- {p}")
                    _c2.markdown("**Cons**")
                    for c in _rs["cons"]:
                        _c2.markdown(f"- {c}")
                    if _rt["flags"]:
                        st.warning("Manipulation signals: " + _rt_detail)
            else:
                st.caption("No reviews tracked for this product yet.")

            bow = _bow_model().predict(store.history(offer["offer_id"]))
            rec = bow["recommendation"]
            cls = "wait" if rec == "WAIT" else "buy"
            p_drop = bow.get("drop_probability", 0.0)
            st.markdown(
                '<div class="pm-section">Buy or wait</div>'
                f'<div class="verdict {cls}">'
                f'<div class="verdict-word">{_esc(rec)}</div>'
                f'<div class="verdict-sub">{_esc(bow["reason"])}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            if p_drop:
                st.progress(min(max(p_drop, 0.0), 1.0),
                            text=f"Model-estimated chance of a lower price soon: {p_drop:.0%}")
            _ex = _bow_model().explain(store.history(offer["offer_id"]))
            if _ex["contributions"]:
                with st.expander("Why this verdict · exact feature contributions"):
                    _cdf = pd.DataFrame(_ex["contributions"]).set_index("feature")
                    if st.session_state.light:
                        # The native bar chart keeps the dark base theme inside
                        # its canvas. Explicit chart styling is needed here;
                        # CSS cannot reach pixels drawn inside a canvas.
                        _contrib_chart = (
                            alt.Chart(_cdf.reset_index())
                            .mark_bar(color="#83c9ff")
                            .encode(x=alt.X("feature:N", sort=None, title=None,
                                            axis=alt.Axis(labelAngle=-90)),
                                    y=alt.Y("contribution:Q", title=None))
                            .properties(height=350, background=_LIGHT["bg"])
                            .configure_axis(labelColor=_LIGHT["sub"],
                                            titleColor=_LIGHT["sub"],
                                            gridColor=_LIGHT["grid"],
                                            domainColor=_LIGHT["border"],
                                            tickColor=_LIGHT["border"])
                            .configure_view(strokeWidth=0))
                        st.altair_chart(_contrib_chart, use_container_width=True,
                                         theme=None)
                    else:
                        st.bar_chart(_cdf["contribution"])
                    for c in _ex["contributions"][:3]:
                        st.markdown(f"- {c['plain']}")
                    st.caption("Linear model, so contribution = learned coefficient x "
                               "feature value (exact, equivalent to SHAP for linear models). "
                               "Positive pushes WAIT, negative pushes BUY.")

            st.markdown('<div class="pm-section">Price-drop alerts</div>',
                        unsafe_allow_html=True)
            st.caption("On-screen checks against sample price history, not "
                       "background notifications. Targets last for this browser session.")
            st.session_state.setdefault("watch", {})
            _watch = st.session_state["watch"]
            _oid = offer["offer_id"]
            _cur = _watch.get(_oid)
            _wc1, _wc2 = st.columns([3, 1])
            _target = _wc1.number_input(
                f"Watch {pick}: alert at or below",
                min_value=0.0, value=float(_cur or int(best or 0)),
                step=50.0, key=f"watch-{_oid}")
            if _wc2.button("Set watch", key=f"set-{_oid}"):
                _watch[_oid] = _target
                _cur = _target
            from pricematch.alerts import scan as _al_scan
            _hists = {o["offer_id"]: [{"ts": str(r.ts), "price": float(r.price)}
                                      for r in store.history(o["offer_id"]).itertuples()]
                      for o in offers}
            _hits = _al_scan(_hists, targets=_watch,
                             current_prices={o["offer_id"]: float(o["price"]) for o in offers})
            if _hits:
                for h in _hits:
                    _site = h["offer_id"].split("::")[-1]
                    _bits = [f"{h['pct_below_avg']}% below its 30-day "
                             f"average of {_money(h['trailing_avg'])}"]
                    if h["new_low"]:
                        _bits.append("new tracked low")
                    if h["hit_target"]:
                        _bits.append(f"hits your {_money(h['target'])} target")
                    st.markdown(
                        f'<div class="offer-card"><div class="offer-left">'
                        f'<div><div class="site">{_esc(_site)} · {_money(h["latest"])}</div>'
                        f'<div class="breakdown">{_esc("; ".join(_bits))}</div>'
                        f"</div></div></div>",
                        unsafe_allow_html=True)
            else:
                st.caption("No meaningful drops on this product right now. A "
                           "drop counts when the latest price is at least 3% "
                           "below its 30-day average and a new tracked low, or "
                           "when it crosses your watch target.")
            if _cur:
                st.caption(f"Watching {pick}: alert at or below {_money(_cur)} "
                           "(kept for this browser session).")
    toolset.close()


def _name_slugs(text: str) -> str:
    """Replace catalog slugs in agent prose with product display names."""
    pattern = r"(?<![\w-])(?:" + "|".join(
        re.escape(cid) for cid in sorted(_CLUSTER_META, key=len, reverse=True)
    ) + r")(?![\w-])"
    return re.sub(pattern, lambda m: _CLUSTER_META[m.group(0)][0], text)


def _split_agent_answer(answer: str) -> dict:
    """Split the grounded answer into presentation layers.

    The pipeline emits one markdown blob: an intro line, bullets, and a
    closing note. For the UI we separate (a) the budget verdict, (b) the
    readable answer bullets and (c) the raw receipts (tool outputs, offer
    evidence). Citability is untouched: every bullet keeps its [n] marker
    and the receipts stay one click away under "How it knows".
    """
    intro, budget, main, receipts = "", None, [], []
    for line in answer.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.startswith("Grounded in "):
            intro = s
        elif s.startswith("Every sentence above cites"):
            continue  # the UI says this in its own words
        elif s.startswith("* Budget check:"):
            budget = s[2:]
        elif s.startswith("* [tool: review_summary]"):
            main.append("Review check: " + s.removeprefix("* [tool: review_summary] "))
        elif s.startswith("* [tool:"):
            receipts.append(s[2:])
        elif s.startswith("* ") and ": listed " in s and "true landed cost" in s:
            receipts.append(s[2:])  # per-offer price evidence
        elif s.startswith("* "):
            main.append(s[2:])
        else:
            main.append(s)
    return {"intro": intro, "budget": budget, "main": main, "receipts": receipts}


with tab_agent:
    st.caption("RAG agent: retrieves the most relevant product and price-intelligence "
               "chunks from the tracked catalog, then answers grounded ONLY in those "
               "chunks, with citations. Fully offline; no API keys.")
    st.session_state.setdefault("aq", "boAt airdopes under 1500")
    q = st.text_area("Ask", key="aq")
    st.markdown('<div class="pm-chip-row">', unsafe_allow_html=True)
    _acols = st.columns(len(_AGENT_EXAMPLES))
    for _col, (_label, _q) in zip(_acols, _AGENT_EXAMPLES):
        _col.button(_label, key=f"achip-{_label}",
                    on_click=_seed, args=("aq", _q))
    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("Go", type="primary"):
        with st.spinner("Thinking - the first answer downloads small local "
                        "models (one time only); later answers are fast."):
            result = _rag().answer(q)
        # Agent prose names products by catalog slug; swap in display names,
        # and remember which products were named so each becomes a one-click
        # jump into Search & compare below.
        _mentioned = [c for c in _CLUSTER_META if re.search(
            r"(?<![\w-])" + re.escape(c) + r"(?![\w-])", result["answer"])]
        parts = _split_agent_answer(_name_slugs(result["answer"]))
        if parts["budget"]:
            fits = "nothing here fits" not in parts["budget"]
            cls = "buy" if fits else "wait"
            body = parts["budget"].removeprefix("Budget check: ")
            st.markdown(
                '<div class="pm-section">Budget check</div>'
                f'<div class="verdict {cls}">'
                f'<div class="verdict-word">{"FITS" if fits else "OVER BUDGET"}</div>'
                f'<div class="verdict-sub">{_esc(body)}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with st.container(border=True):
            for bullet in parts["main"]:
                st.markdown(f"- {bullet}" if not bullet.startswith("**") else bullet)
        if _mentioned:
            st.caption("Open in Search & compare:")
            for _r in range(0, min(len(_mentioned), 8), 4):
                _ocols = st.columns(min(len(_mentioned) - _r, 4))
                for _col, _cid in zip(_ocols, _mentioned[_r:_r + 4]):
                    # a real navigation, not a session pin: the selected tab
                    # is client-side and can survive a rerun, so a pure pin
                    # can land invisibly on a hidden tab
                    _col.link_button(_CLUSTER_META[_cid][0], _share_url(_cid),
                                     key=f"open-{_cid}", use_container_width=True)
        st.caption(f"retriever: {result['retriever']} · generator: {result['generator']} · "
                   f"{len(result['retrieved'])} chunks cited · every [n] above maps to a "
                   "source chunk below - nothing here is invented.")
        with st.expander("How it knows - sources, tool receipts and agent plan"):
            if result.get("plan"):
                st.markdown("**Agent plan** - the tools it decided to call, in order")
                for i, line in enumerate(result["plan"], start=1):
                    st.markdown(f"{i}. {_name_slugs(line)}")
            if parts["receipts"]:
                st.markdown("**Tool receipts & offer evidence**")
                for r in parts["receipts"]:
                    st.markdown(f"- {_name_slugs(r)}")
            st.markdown("**Retrieved context** - the chunks the answer cites")
            for i, (doc, score) in enumerate(result["retrieved"], start=1):
                st.markdown(f"**[{i}]** `{doc.meta.get('type')}` · score {score:.2f}")
                st.markdown(_name_slugs(doc.text))

@st.cache_resource
def _photo_searcher():
    from pricematch.photo import PhotoSearcher
    ps = PhotoSearcher()
    # only clusters that actually exist in the catalog DB: stale photo files
    # left behind by an old patch must never become matchable "products"
    known = set(store.offers()["cluster_id"])
    ps.photos = {cid: paths for cid, paths in ps.photos.items() if cid in known}
    return ps


with tab_photo:
    st.caption("Snap or upload a product photo. CLIP embeds it offline and matches it "
               "against the tracked catalog. When the match is decisive you get the "
               "product's true landed cost and verdict; when it isn't, you get visually "
               "similar products instead - never a confident price for the wrong product.")
    shot = st.file_uploader("Product photo", type=["jpg", "jpeg", "png"])
    _ps = _photo_searcher()
    _reason = _ps.unavailable_reason
    if _reason:
        st.info(f"Photo search is unavailable here: {_reason}. "
                "The rest of the demo works without it.")
    elif shot:
        from PIL import Image
        _img = Image.open(shot).convert("RGB")
        st.image(_img, width=220)
        with st.spinner("Matching against the catalog…"):
            _out = _ps.match(_img)
        if _out["matches"] and _out["matches"][0]["confident"]:
            _hit = _out["matches"][0]
            _cid = _hit["cluster_id"]
            st.success(f"Matched: {_cid.replace('-', ' ')} "
                       f"(confidence {_hit['confidence']:.0%}, "
                       f"similarity {_hit['score']:.2f})")
            _muri = _photo_data_uri(_cid, px=200)
            if _muri:
                st.markdown(
                    f'<div class="pm-prod"><img src="{_muri}" alt="">'
                    f'<div><div class="nm">{_esc(_CLUSTER_META.get(_cid, (_cid, ""))[0])}</div>'
                    f'<div class="ct">catalog reference photo for the matched product</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            if len(_out["matches"]) > 1:
                st.caption("runners-up: " + ", ".join(
                    f"{m['cluster_id']} ({m['score']:.2f})" for m in _out["matches"][1:]))
            _pts2 = Toolset(DB_PATH)
            _ranked = _pts2.landed_costs(_cid)
            _offers2 = _pts2.cluster_offers(_cid)
            if _ranked:
                _b = _ranked[0]
                _bow2 = _bow_model().predict(store.history(_offers2[0]["offer_id"]))
                st.markdown(
                    f'<div class="pm-section">Best true landed cost</div>'
                    f'<div class="landed"><span class="amt">{_money(_b["landed_cost"])}</span>'
                    f'<span class="lbl">at {_esc(_b["site"])} · verdict '
                    f'{_esc(_bow2["recommendation"])} · {_esc(_bow2["reason"])}</span></div>',
                    unsafe_allow_html=True,
                )
            _pts2.close()
        else:
            st.warning("No confident match - this photo is too different from "
                       "anything in the tracked catalog to name the product.")
            if _out["matches"]:
                st.markdown('<div class="pm-section">Visually similar products</div>',
                            unsafe_allow_html=True)
                for _m in _out["matches"]:
                    st.markdown(
                        f"- {_m['cluster_id'].replace('-', ' ')} "
                        f"· similarity {_m['score']:.2f}")
                st.caption("Prices and verdicts appear only for a confident match - "
                           "a wrong label is worse than none.")


with tab_compare:
    st.caption("Pick up to three tracked products and Cipher lines up their "
               "cheapest offers, true landed costs and buy-or-wait verdicts "
               "side by side.")
    _cmp_ts = Toolset(DB_PATH)
    _cmp_picks = st.multiselect(
        "Products to compare",
        options=sorted(_CLUSTER_META, key=lambda c: _CLUSTER_META[c][0]),
        format_func=lambda c: _CLUSTER_META[c][0],
        max_selections=3,
        placeholder="Choose 2 or 3 products")
    if len(_cmp_picks) >= 2:
        _cards = []
        for _cid in _cmp_picks:
            _rk = _cmp_ts.landed_costs(_cid)
            _ofs = _cmp_ts.cluster_offers(_cid)
            _b = _rk[0] if _rk else None
            _bo = next((o for o in _ofs if _b and o["site"] == _b["site"]),
                       _ofs[0] if _ofs else None)
            _bw = _cmp_ts.buy_or_wait(_bo["offer_id"]) if _bo else None
            _chk = _cmp_ts.price_check(_bo["offer_id"]) if _bo else None
            _cards.append({"cid": _cid, "best": _b, "bow": _bw, "chk": _chk})
        _min_landed = min(c["best"]["landed_cost"] for c in _cards if c["best"])
        _ccols = st.columns(len(_cards))
        for _col, _c in zip(_ccols, _cards):
            with _col:
                with st.container(border=True):
                    _cu = _photo_data_uri(_c["cid"])
                    if _cu:
                        st.image(_cu, width=140)
                    st.markdown(f"**{_esc(_CLUSTER_META[_c['cid']][0])}**")
                    _b = _c["best"]
                    if _b:
                        _tag = (" · LOWEST COST"
                                if _b["landed_cost"] == _min_landed else "")
                        st.markdown(
                            f'<div class="landed"><span class="amt">'
                            f'{_money(_b["landed_cost"])}</span>'
                            f'<span class="lbl">true landed cost at '
                            f'{_esc(_b["site"])}{_tag}</span></div>',
                            unsafe_allow_html=True)
                        st.markdown(
                            f"- Listed {_money(_b['price'])} · "
                            f"{'free shipping' if not _b['shipping'] else '+' + _money(_b['shipping']) + ' shipping'}"
                            + (f" · coupon saves {_money(_b['coupon_value'])}"
                               if _b["coupon_value"] else ""))
                        st.markdown(f"- Returns: {_esc(_b['risk'])}")
                    if _c["chk"]:
                        for flag in _c["chk"]["flags"]:
                            _lb, _cl = _FLAG_LABELS.get(
                                flag, (flag.replace("_", " "), "pill-gray"))
                            st.markdown(f'<span class="pill {_cl}">{_esc(_lb)}</span>',
                                        unsafe_allow_html=True)
                    if _c["bow"]:
                        st.markdown(
                            f"- Verdict: **{_esc(_c['bow']['recommendation'])}** "
                            f"- {_esc(_c['bow']['reason'])}")
                    st.link_button("Open in Search & compare", _share_url(_c["cid"]),
                                   key=f"cmp-{_c['cid']}")
    elif _cmp_picks:
        st.caption("Pick at least one more product to compare.")
    _cmp_ts.close()

store.close()
