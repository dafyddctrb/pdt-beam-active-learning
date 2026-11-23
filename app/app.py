"""Local web app for the beam PDT.  Start with:  streamlit run app/app.py"""
import dataclasses
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from app import docs, load
from app.figures import (COLORS, FONT,  # noqa: E402
                         posterior_caption, posterior_grid_fig, split_line, study_caption, study_fig)  # noqa: E402
from app.worker import create_run, read_json, start_run, stop_run  # noqa: E402
from case_study_config import CaseStudyConfig  # noqa: E402

CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
METHODS = {"seq_pm": "Seq PM", "seq_srg": "Seq SRG", "batch_pm": "Batch PM", "batch_srg": "Batch SRG"}
FIELDS = {f.name: f for f in dataclasses.fields(CaseStudyConfig)}
ACQ = ["bus_u_learning", "bus_u_learning_2nd"]
CONV = ["pf_stability", "min_u", "pf_bound", "beta_stability", "beta_bound", "combined", "seeded_delta"]
CONV_HELP = ("Learning-function based: stop when every candidate has U ≥ 2 (Echard et al. 2011). "
             "Stability of Pf / β: stop when the estimate changes by less than the tolerance between "
             "iterations. Bounds on Pf / β: stop when the bounds from the surrogate's ± 2σ band are "
             "within the tolerance. Combined: stability and bounds on β, twice in a row "
             "(Moustapha et al. 2022).")

# Widget spec per config field: ("slider", label, lo, hi, step, fmt, scale) | ("select", label, options)
# Sliders show value/scale (e.g. E in GPa); the config stores value*scale.
SPEC = {
    "beam_type": ("select", "Structural system", ["cantilever", "two_sided"]),
    "l": ("slider", "Span l [m]", 1.0, 30.0, 0.5, "%.1f", 1),
    "V": ("slider", "Load V [kN]", 0.5, 20.0, 0.5, "%.1f", 1),
    "b": ("slider", "Section width b [m]", 0.1, 3.0, 0.05, "%.2f", 1),
    "h": ("slider", "Section height h [m]", 0.05, 1.0, 0.01, "%.2f", 1),
    "true_E": ("slider", "True E [GPa]", 50.0, 400.0, 5.0, "%.0f", 1e6),
    "true_lambda": ("slider", "True λ [%/year]", 0.1, 3.0, 0.05, "%.2f", 1),
    "prior_E_factor": ("slider", "Prior mean of E / true E", 0.5, 1.5, 0.01, "%.2f", 1),
    "prior_E_cov": ("slider", "Prior CoV of E", 0.01, 0.5, 0.01, "%.2f", 1),
    "prior_lambda_factor": ("slider", "Prior mean of λ / true λ", 0.5, 1.5, 0.01, "%.2f", 1),
    "prior_lambda_cov": ("slider", "Prior CoV of λ", 0.01, 0.5, 0.01, "%.2f", 1),
    "t_start": ("slider", "First year", 0, 10, 1, "%d", 1),
    "t_end": ("slider", "Last year", 5, 50, 1, "%d", 1),
    "t_step_displacement": ("slider", "Displacement measured every … years", 1, 10, 1, "%d", 1),
    "t_start_deterioration": ("slider", "First deterioration measurement [year]", 1, 30, 1, "%d", 1),
    "t_step_deterioration": ("slider", "Deterioration measured every … years", 1, 20, 1, "%d", 1),
    "t_end_deterioration": ("slider", "Last deterioration measurement [year]", 1, 50, 1, "%d", 1),
    "displacement_noise_rel": ("slider", "Displacement noise (relative)", 0.0, 0.1, 0.001, "%.3f", 1),
    "displacement_sigma": ("slider", "Likelihood σ of displacement [m]", 0.001, 0.05, 0.001, "%.3f", 1),
    "loss_noise_rel": ("slider", "Deterioration noise (relative)", 0.0, 0.3, 0.005, "%.3f", 1),
    "sigma_lambda": ("slider", "Likelihood σ of λ [%/year]", 0.01, 1.0, 0.01, "%.2f", 1),
    "n_samples_pm": ("slider", "Samples per level, physics model", 50, 2000, 50, "%d", 1),
    "n_samples_srg": ("slider", "Samples per level, surrogate", 100, 10000, 100, "%d", 1),
    "p0": ("slider", "SuS conditional probability p0", 0.05, 0.5, 0.05, "%.2f", 1),
    "acquisition_function": ("select", "Learning function", ACQ),
    "convergence_method": ("select", "Stopping criterion", CONV),
    "convergence_criteria": ("slider", "Convergence tolerance (relative change of Pf)", 0.005, 0.5, 0.005, "%.3f", 1),
    "min_u_threshold": ("slider", "Threshold on min U", 0.5, 5.0, 0.1, "%.1f", 1),
    "seeded_delta_eps": ("slider", "Tolerance ε (absolute change of the failure fraction)", 0.005, 0.5, 0.005, "%.3f", 1),
    "n_iterations": ("slider", "Max AL iterations per update", 1, 100, 1, "%d", 1),
    "min_train_points": ("slider", "Minimum training-set size", 5, 100, 1, "%d", 1),
    "epochs": ("slider", "GP training epochs", 10, 1000, 10, "%d", 1),
    "lr": ("select", "GP learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),
}

# ---------------------------------------------------------------- look
st.set_page_config(page_title="Beam PDT", page_icon=os.path.join(os.path.dirname(__file__), "icon.png"), layout="wide")
st.markdown(f"""<style>
html, body, .stApp, div, p, label, li, h1, h2, h3, h4, h5, h6, input, textarea, button, td, th, summary
  {{ font-family: {FONT} !important; }}
.stApp {{ background: #fff; color: #000; }}
.stApp p, .stApp li {{ line-height: 1.55; }}
.stApp p {{ text-align: justify; hyphens: auto; -webkit-hyphens: auto; }}
h1 {{ font-weight: 400 !important; font-size: clamp(1.6rem, 1.1rem + 1.4vw, 2.5rem) !important; letter-spacing: -0.01em; padding-bottom: 0 !important; }}
h4, h5 {{ font-weight: 400 !important; font-style: italic; font-size: 1.2rem !important; padding: 0.4rem 0 0.2rem !important; }}
h5 {{ font-size: 1.05rem !important; }}
.block-head {{ font-size: 1.6rem; border-top: 1px solid #000; padding-top: 0.7rem; margin: 2.2rem 0 0.4rem; }}
.block-head span {{ display: inline-block; width: 0.75em; height: 0.75em; margin-right: 0.45em; vertical-align: 0.02em; }}
.abstract-head {{ font-weight: 700; font-size: 1.05rem; margin: 0.6rem 0 0.2rem; }}
.abstract {{ font-size: clamp(0.98rem, 0.9rem + 0.25vw, 1.15rem); margin: 0; }}
.abstract:last-child {{ margin-bottom: 1.6rem; }}
.fig-cap {{ font-size: 0.9rem; color: #4d4d4d; margin: -0.4rem 0 1.2rem; line-height: 1.45; }}
.fig-cap b {{ color: #000; font-weight: 700; }}
/* sections read like a paper: a full-width rule and a heading, no box */
[data-testid="stExpander"] details {{ border: 0; border-top: 1px solid #000; border-radius: 0; }}
[data-testid="stExpander"] summary {{ padding-left: 0; padding-right: 0; }}
[data-testid="stExpander"] summary p {{ font-size: clamp(1.25rem, 1rem + 0.6vw, 1.6rem); }}
[data-testid="stExpander"] summary:focus-visible {{ outline: 2px solid #000; outline-offset: 2px; }}
[data-testid="stExpanderDetails"] {{ padding-left: 0; padding-right: 0; }}
/* notes column: the documentation of the section, beside it */
[data-testid="stColumn"]:has(.pdt-notes):not(:has([data-testid="stColumn"]))
  {{ border-left: 1px solid #000; padding-left: 1rem; max-height: 40rem; overflow-y: auto; }}
[data-testid="stColumn"]:has(.pdt-notes):not(:has([data-testid="stColumn"])) p,
[data-testid="stColumn"]:has(.pdt-notes):not(:has([data-testid="stColumn"])) li
  {{ font-size: 0.9rem; line-height: 1.45; }}
.pdt-notes {{ font-style: italic; font-size: 1.3rem !important; line-height: 1.25 !important; margin: 0.4rem 0 0.4rem; }}
[data-testid="stCheckbox"] label p {{ white-space: nowrap; }}
/* below 1200 px the notes no longer fit beside the content: they move underneath it, full width */
@media (max-width: 1200px) {{
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .pdt-notes) {{ flex-wrap: wrap; }}
  [data-testid="stColumn"]:has(.pdt-notes):not(:has([data-testid="stColumn"])) {{
    flex: 1 1 100% !important; width: 100% !important; min-width: 100% !important;
    border-left: 0; border-top: 1px solid #b3b3b3; padding: 0.4rem 0 0; max-height: 22rem; }}
}}
</style>""", unsafe_allow_html=True)



FIG = [0]          # figure counter, restarts on every rerun


def caption(text, where=st):
    FIG[0] += 1
    where.markdown(f'<p class="fig-cap"><b>Figure {FIG[0]}.</b> {text}</p>', unsafe_allow_html=True)


def show(fig, where=st, cap=None):
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", legend_title_text="")
    where.plotly_chart(fig, use_container_width=True, theme=None)
    if cap:
        caption(cap, where)


def render_topic(key):
    """Body of one documentation topic (app/docs.py), as in section 0 and in the notes column."""
    t = docs.TOPICS[key]
    st.markdown("*To be written.*" if t["status"] == "todo" else t["long"].strip())


NOTES_FRAC = 0.26   # share of the row taken by the notes column


def notes_switch(sec, c=None):
    """Right-aligned button at the top of a section that shows or hides its notes column
    (in column c if given, else in a row of its own)."""
    if c is None:
        _, c = st.columns([6, 1])
    key = f"notes_{sec}"
    on = st.session_state.setdefault(key, True)
    c.button("Hide notes  →" if on else "←  Show notes", key=f"btn_{key}", use_container_width=True,
             on_click=lambda: st.session_state.update({key: not on}),
             help="Shows or hides the documentation of this section in a column on its right.")


def noted(spec, *topics, sec):
    """Columns for a (sub)section, as st.columns(spec), with the documentation of `topics` in an
    extra column on the right while the notes of section `sec` are on. Returns the content columns."""
    if not st.session_state.get(f"notes_{sec}", True):
        return st.columns(spec, gap="large") if len(spec) > 1 else [st.container()]
    cols = st.columns(list(spec) + [sum(spec) * NOTES_FRAC / (1 - NOTES_FRAC)], gap="large")
    with cols[-1]:
        for k in topics:
            st.markdown(f'<p class="pdt-notes">{docs.TOPICS[k]["title"]}</p>', unsafe_allow_html=True)
            render_topic(k)
    return cols[:-1]


# ---------------------------------------------------------------- config state
def _has_value(key) -> bool:
    """True if the session holds a value for `key`. A widget key can be registered without
    a value (`key in st.session_state` is then True, but reading it raises KeyError)."""
    try:
        st.session_state[key]
        return True
    except KeyError:
        return False


def set_config(cfg: CaseStudyConfig, only_missing: bool = False):
    for k, v in cfg.to_dict().items():
        if k not in SPEC or (only_missing and _has_value(f"w_{k}")):
            continue
        spec = SPEC[k]
        if spec[0] == "slider":
            v = int(v) if FIELDS[k].type == "int" else float(v) / spec[6]
        st.session_state[f"w_{k}"] = v


def current_config() -> CaseStudyConfig:
    d = {}
    for k, spec in SPEC.items():
        v = st.session_state[f"w_{k}"]
        if spec[0] == "slider" and FIELDS[k].type == "float":
            v = float(f"{v * spec[6]:.10g}")          # strip slider float noise
        d[k] = v
    return CaseStudyConfig.from_dict(d)


def widget(name, where=st):
    kind, label, *a = SPEC[name]
    if name == "V":       # cantilever: distributed load q; simply supported: centre point load
        label = ("Distributed load q [kN/m]" if st.session_state["w_beam_type"] == "cantilever"
                 else "Point load at midspan V [kN]")
    key, v = f"w_{name}", st.session_state[f"w_{name}"]
    if kind == "select":
        opts = a[0] if v in a[0] else a[0] + [v]
        if name == "lr":
            where.select_slider(label, opts, key=key, format_func=lambda x: f"{x:g}", help=docs.tip(name))
        else:
            names = {"cantilever": "Cantilever", "two_sided": "Simply supported", **load.ACQ_NAMES, **load.CONV_NAMES}
            where.selectbox(label, opts, key=key, format_func=lambda x: names.get(x, x), help=docs.tip(name))
    else:
        lo, hi, step, fmt, _ = a
        where.slider(label, min(lo, v), max(hi, v), step=step, format=fmt, key=key, help=docs.tip(name))


if "w_beam_type" not in st.session_state:
    set_config(CaseStudyConfig())
else:
    # a browser session opened before a setting was added keeps its values; the new
    # setting starts at its default
    set_config(CaseStudyConfig(), only_missing=True)
# Streamlit forgets the value of a widget that is not drawn in a rerun; parameters that are
# hidden (e.g. min-U threshold when another convergence method is chosen) must survive.
for _k in SPEC:
    st.session_state[f"w_{_k}"] = st.session_state[f"w_{_k}"]

# Settings the synthetic measurements depend on (beam, truth, schedule, noise, likelihood σ).
MEAS_FIELDS = ("beam_type", "l", "V", "b", "h", "true_E", "true_lambda", "t_start", "t_end",
               "t_step_displacement", "t_start_deterioration", "t_step_deterioration",
               "t_end_deterioration", "displacement_noise_rel", "displacement_sigma", "loss_noise_rel", "sigma_lambda")


def meas_stale() -> bool:
    old, new = st.session_state.meas_cfg, current_config()
    return any(getattr(old, f) != getattr(new, f) for f in MEAS_FIELDS)


def running_run():
    return next((r for r in load.list_runs() if r["state"] == "running"), None)


# ---------------------------------------------------------------- figures for the configuration
def _pdf(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def plot_prior(cfg):
    muE, sE = cfg.true_E * cfg.prior_E_factor / 1e6, cfg.true_E * cfg.prior_E_factor * cfg.prior_E_cov / 1e6
    muL, sL = cfg.true_lambda * cfg.prior_lambda_factor, cfg.true_lambda * cfg.prior_lambda_factor * cfg.prior_lambda_cov
    tE, tL = cfg.true_E / 1e6, cfg.true_lambda
    e = np.linspace(min(muE - 4 * sE, tE - sE), max(muE + 4 * sE, tE + sE), 150)
    lam = np.linspace(min(muL - 4 * sL, tL - sL), max(muL + 4 * sL, tL + sL), 150)
    pE, pL = _pdf(e, muE, sE), _pdf(lam, muL, sL)
    fig = make_subplots(rows=2, cols=2, column_widths=[0.8, 0.2], row_heights=[0.2, 0.8],
                        shared_xaxes="columns", shared_yaxes="rows", horizontal_spacing=0.02, vertical_spacing=0.02)
    fig.add_trace(go.Contour(x=e, y=lam, z=np.outer(pL, pE), showscale=False, ncontours=8,
                             contours=dict(coloring="lines"), colorscale=[[0, "#999"], [1, "#000"]],
                             line=dict(width=1), hoverinfo="skip"), 2, 1)
    fig.add_trace(go.Scatter(x=[tE], y=[tL], mode="markers", marker=dict(symbol="star", size=13, color="#D55E00"),
                             name="true value"), 2, 1)
    fig.add_trace(go.Scatter(x=[muE], y=[muL], mode="markers", marker=dict(symbol="cross-thin", size=12,
                             line=dict(width=2, color="black")), name="prior mean"), 2, 1)
    fig.add_trace(go.Scatter(x=e, y=pE, line=dict(color="black"), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=pL, y=lam, line=dict(color="black"), showlegend=False), 2, 2)
    fig.add_vline(x=tE, line=dict(color="#D55E00", dash="dot"), row=1, col=1)
    fig.add_hline(y=tL, line=dict(color="#D55E00", dash="dot"), row=2, col=2)
    fig.update_xaxes(title_text="E [GPa]", row=2, col=1)
    fig.update_yaxes(title_text="λ [%/year]", row=2, col=1)
    for r, c in ((1, 1), (2, 2), (1, 2)):
        fig.update_xaxes(showticklabels=False, row=r, col=c)
        fig.update_yaxes(showticklabels=False, row=r, col=c)
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)
    fig.update_layout(height=370, margin=dict(l=10, r=10, t=10, b=10), legend=dict(x=0.82, y=1, xanchor="left"))
    return fig


def plot_schedule(cfg):
    fig = go.Figure()
    for y, times in (("displacement", cfg.displacement_times), ("deterioration", cfg.deterioration_times)):
        fig.add_trace(go.Scatter(x=times, y=[y] * len(times), mode="markers", showlegend=False,
                                 marker=dict(symbol="line-ns", size=14, line=dict(width=2, color="black"))))
    fig.update_xaxes(title_text="t [years]", range=[cfg.t_start - 0.5, cfg.t_end + 0.5])
    fig.update_layout(height=170, margin=dict(l=10, r=10, t=10, b=10), title=None)
    return fig


def _var(sym, value, unit):
    return f'<tspan font-style="italic">{sym}</tspan> = {value} {unit}'


def cantilever_svg(cfg) -> str:
    """Upright cantilever as in Figure 4a of the paper: clamped at the bottom, the line load acting
    sideways along its height, the tip deflecting by δ_max. Not to scale."""
    y0, y1 = 262, 42                 # beam drawn from the base y0 up to the tip y1
    xb, t = 190, 6 + 18 * min(cfg.h, 1.0)    # left edge and drawn thickness of the beam
    xr = xb + t
    s = ['<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#000"/></marker></defs>']
    # clamped base
    s.append(f'<line x1="{xb - 45}" y1="{y0}" x2="{xr + 45}" y2="{y0}" stroke="#000" stroke-width="2"/>')
    s += [f'<line x1="{x}" y1="{y0}" x2="{x - 10}" y2="{y0 + 10}" stroke="#000"/>' for x in range(xb - 38, int(xr) + 46, 10)]
    # line load: arrows from the left onto the beam, tails joined
    xl = xb - 70
    ys = np.linspace(y1 + 4, y0 - 8, 10)
    s.append(f'<line x1="{xl}" y1="{ys[0]}" x2="{xl}" y2="{ys[-1]}" stroke="#000"/>')
    s += [f'<line x1="{xl}" y1="{y:.1f}" x2="{xb - 2}" y2="{y:.1f}" stroke="#000" marker-end="url(#ah)"/>' for y in ys]
    s.append(f'<text x="{xl - 10}" y="{(y0 + y1) / 2}" text-anchor="end">{_var("q", f"{cfg.V:g}", "kN/m")}</text>')
    # beam and deflected shape (uniform-load deflection, normalised)
    s.append(f'<rect x="{xb}" y="{y1}" width="{t}" height="{y0 - y1}" fill="#fff" stroke="#000" stroke-width="1.5"/>')
    xs = np.linspace(0, 1, 60)
    w, amp = (6 * xs**2 - 4 * xs**3 + xs**4) / 3, 42
    pts = " ".join(f"{xr + amp * b:.1f},{y0 - (y0 - y1) * a:.1f}" for a, b in zip(xs, w))
    s.append(f'<polyline points="{pts}" fill="none" stroke="#000" stroke-dasharray="5 4" stroke-width="1"/>')
    s.append(f'<line x1="{xr + 3}" y1="{y1 - 10}" x2="{xr + amp}" y2="{y1 - 10}" stroke="#000" '
             f'marker-end="url(#ah)"/>')
    s.append(f'<text x="{xr + amp + 8}" y="{y1 - 5}"><tspan font-style="italic">δ</tspan>'
             f'<tspan font-size="11" dy="4">max</tspan></text>')
    # height dimension
    xd = xr + 95
    s.append(f'<line x1="{xd}" y1="{y0}" x2="{xd}" y2="{y1}" stroke="#000"/>')
    s += [f'<line x1="{xd - 5}" y1="{y + 5}" x2="{xd + 5}" y2="{y - 5}" stroke="#000" stroke-width="1.5"/>' for y in (y0, y1)]
    s += [f'<line x1="{xd - 9}" y1="{y}" x2="{xd + 9}" y2="{y}" stroke="#000" stroke-width="0.6"/>' for y in (y0, y1)]
    s.append(f'<text x="{xd + 10}" y="{(y0 + y1) / 2 + 5}">{_var("l", f"{cfg.l:g}", "m")}</text>')
    # cross-section seen from above: the depth h lies in the load direction (horizontal here),
    # the width b points into the page
    sc = 90 / max(cfg.b, cfg.h)
    sw, sh, cx, cy = max(cfg.h * sc, 2), cfg.b * sc, 455, (y0 + y1) / 2
    s.append(f'<rect x="{cx - sw / 2}" y="{cy - sh / 2}" width="{sw}" height="{sh}" fill="#fff" stroke="#000" stroke-width="1.5"/>')
    s.append(f'<text x="{cx}" y="{cy + sh / 2 + 20}" text-anchor="middle">{_var("h", f"{cfg.h:.2f}", "m")}</text>')
    s.append(f'<text x="{cx + sw / 2 + 8}" y="{cy + 5}">{_var("b", f"{cfg.b:.2f}", "m")}</text>')
    s.append(f'<text x="{cx}" y="{cy - sh / 2 - 10}" text-anchor="middle" font-size="12" fill="#555">section</text>')
    return (f'<svg viewBox="0 0 560 285" width="100%" role="img" aria-label="Upright cantilever with a line load" '
            f'style="font-family:{FONT.replace(chr(34), chr(39))};font-size:15px;max-width:560px">{"".join(s)}</svg>')


def beam_svg(cfg) -> str:
    """Engineering sketch of the structural system (not to scale), redrawn from the config."""
    if cfg.beam_type == "cantilever":
        return cantilever_svg(cfg)
    x0, x1, yc = 70, 430, 100
    t = 6 + 34 * min(cfg.h, 1.0)
    top, bot = yc - t / 2, yc + t / 2
    s = ['<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#000"/></marker></defs>']
    xs = np.linspace(0, 1, 60)
    if cfg.beam_type == "cantilever":
        w, amp = (6 * xs**2 - 4 * xs**3 + xs**4) / 3, 30        # uniform-load deflection, normalised
        s.append(f'<line x1="{x0}" y1="45" x2="{x0}" y2="155" stroke="#000" stroke-width="2"/>')
        s += [f'<line x1="{x0}" y1="{y}" x2="{x0 - 12}" y2="{y + 12}" stroke="#000"/>' for y in range(47, 150, 10)]
        ql = top - 30
        s.append(f'<line x1="{x0}" y1="{ql}" x2="{x1}" y2="{ql}" stroke="#000"/>')
        s += [f'<line x1="{x}" y1="{ql}" x2="{x}" y2="{top - 2}" stroke="#000" marker-end="url(#ah)"/>'
              for x in np.linspace(x0 + 12, x1, 11)]
        s.append(f'<text x="{(x0 + x1) / 2}" y="{ql - 9}" text-anchor="middle">{_var("q", f"{cfg.V:g}", "kN/m")}</text>')
    else:
        w, amp = np.where(xs <= 0.5, 3 * xs - 4 * xs**3, 3 * (1 - xs) - 4 * (1 - xs)**3), 25   # centre load
        for x, roller in ((x0, False), (x1, True)):
            g = bot + 18 + (8 if roller else 0)
            s.append(f'<path d="M{x},{bot} L{x - 11},{bot + 18} L{x + 11},{bot + 18} z" fill="#fff" stroke="#000"/>')
            if roller:
                s += [f'<circle cx="{x + dx}" cy="{bot + 22}" r="3.5" fill="#fff" stroke="#000"/>' for dx in (-6, 6)]
            s.append(f'<line x1="{x - 18}" y1="{g}" x2="{x + 18}" y2="{g}" stroke="#000"/>')
            s += [f'<line x1="{x + dx}" y1="{g}" x2="{x + dx - 7}" y2="{g + 7}" stroke="#000"/>' for dx in range(-14, 19, 7)]
        xm = (x0 + x1) / 2
        s.append(f'<line x1="{xm}" y1="28" x2="{xm}" y2="{top - 2}" stroke="#000" stroke-width="1.5" marker-end="url(#ah)"/>')
        s.append(f'<text x="{xm + 10}" y="40">{_var("V", f"{cfg.V:g}", "kN")}</text>')
    s.append(f'<rect x="{x0}" y="{top}" width="{x1 - x0}" height="{t}" fill="#fff" stroke="#000" stroke-width="1.5"/>')
    pts = " ".join(f"{x0 + (x1 - x0) * a:.1f},{bot + amp * b:.1f}" for a, b in zip(xs, w))
    s.append(f'<polyline points="{pts}" fill="none" stroke="#000" stroke-dasharray="5 4" stroke-width="1"/>')
    # span dimension
    yd = 182
    s.append(f'<line x1="{x0}" y1="{yd}" x2="{x1}" y2="{yd}" stroke="#000"/>')
    s += [f'<line x1="{x - 5}" y1="{yd + 5}" x2="{x + 5}" y2="{yd - 5}" stroke="#000" stroke-width="1.5"/>' for x in (x0, x1)]
    s += [f'<line x1="{x}" y1="{yd - 9}" x2="{x}" y2="{yd + 9}" stroke="#000" stroke-width="0.6"/>' for x in (x0, x1)]
    s.append(f'<text x="{(x0 + x1) / 2}" y="{yd + 20}" text-anchor="middle">{_var("l", f"{cfg.l:g}", "m")}</text>')
    # cross-section b x h
    sc = 64 / max(cfg.b, cfg.h)
    bw, bh, cx = cfg.b * sc, max(cfg.h * sc, 2), 505
    s.append(f'<rect x="{cx - bw / 2}" y="{yc - bh / 2}" width="{bw}" height="{bh}" fill="#fff" stroke="#000" stroke-width="1.5"/>')
    s.append(f'<text x="{cx}" y="{yc - bh / 2 - 9}" text-anchor="middle">{_var("h", f"{cfg.h:.2f}", "m")}</text>')
    s.append(f'<text x="{cx}" y="{yc + bh / 2 + 21}" text-anchor="middle">{_var("b", f"{cfg.b:.2f}", "m")}</text>')
    label = ("Cantilever with distributed load" if cfg.beam_type == "cantilever"
             else "Simply supported beam with a point load at midspan")
    return (f'<svg viewBox="0 0 560 215" width="100%" role="img" aria-label="{label}" '
            f'style="font-family:{FONT.replace(chr(34), chr(39))};font-size:15px;max-width:640px">{"".join(s)}</svg>')


# ================================================================= page
st.title("Probabilistic digital twin of a deteriorating beam")
st.markdown('<p class="abstract-head">Abstract</p>', unsafe_allow_html=True)
st.markdown('<p class="abstract">This repository accompanies the first numerical investigation in <i>Bayesian Data '
            'Integration and Model Updating in Probabilistic Digital Twins</i>: a deteriorating steel cantilever '
            'beam.</p>'
            '<p class="abstract">The example combines deterioration measurements (<b>property data</b>) and '
            'displacement measurements (<b>behavior data</b>) to update uncertainty in Young\'s modulus <i>E</i> '
            'and the deterioration rate λ. It implements sequential and batch updating, each with either a '
            'physics-based model or a Gaussian-process surrogate refined through active learning.</p>'
            '<p class="abstract">The analytical beam model and two uncertain parameters make this a small example '
            'that can be run locally, with numerical integration providing a reference posterior. The repository '
            'contains the implementation, configuration files and a local browser-based app. The paper provides '
            'the methodological details.</p>',
            unsafe_allow_html=True)

# ---------------------------------------------------------------- 0 documentation
with st.expander("Introduction"):
    st.markdown('<p class="fig-cap">General information. What belongs to one section is in the notes '
                'beside it (show or hide them with the button at the top right of each section), and the ⓘ tips '
                'next to the controls give a short version.</p>', unsafe_allow_html=True)
    for key in docs.GENERAL:
        st.markdown(f"#### {docs.TOPICS[key]['title']}")
        render_topic(key)

# ---------------------------------------------------------------- 1 configuration
with st.expander("1  Numerical investigation", expanded=True):
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CONFIG_DIR, "*.json")))
    c1, c2, c3, c4, c5 = st.columns([2, 1, 2, 1, 1], vertical_alignment="bottom")
    notes_switch("1", c5)
    pick = c1.selectbox("Configuration file", ["(defaults)"] + files)
    if c2.button("Load"):
        set_config(CaseStudyConfig() if pick == "(defaults)" else
                   CaseStudyConfig().updated(**json.load(open(os.path.join(CONFIG_DIR, pick)))))
        st.rerun()
    name = c3.text_input("Save as", "my_config")
    if c4.button("Save"):
        with open(os.path.join(CONFIG_DIR, f"{name}.json"), "w") as f:
            json.dump(current_config().to_dict(), f, indent=2)
        st.success(f"Saved configs/{name}.json")

    st.markdown("#### Structure and ground truth")
    left, right = noted([1, 1.3], "model", sec="1")
    with left:
        a, b = st.columns(2)
        for i, n in enumerate(("beam_type", "V", "l", "b", "h", "true_E", "true_lambda")):
            widget(n, (a, b)[i % 2])
    with right:
        st.markdown(beam_svg(current_config()), unsafe_allow_html=True)
        caption("Structural system of the physical twin, not to scale. The dashed line is the deflected shape, "
                "and deterioration reduces the section and with it the stiffness over time.")

    st.markdown("#### Prior of the digital twin")
    left, right = noted([1, 1.3], "prior", sec="1")
    with left:
        for n in ("prior_E_factor", "prior_E_cov", "prior_lambda_factor", "prior_lambda_cov"):
            widget(n)
    with right:
        show(plot_prior(current_config()), cap="Joint prior of the stiffness E and the deterioration rate λ "
             "(independent normals), with the marginals on the margins. The star marks the true value.")

    st.markdown("#### Measurements")
    left, right = noted([1, 1.3], "measurements", sec="1")
    with left:
        a, b = st.columns(2)
        for i, n in enumerate(("t_start", "t_end", "t_step_displacement", "t_step_deterioration",
                               "t_start_deterioration", "t_end_deterioration")):
            widget(n, (a, b)[i % 2])
        for i, n in enumerate(("displacement_noise_rel", "loss_noise_rel", "displacement_sigma", "sigma_lambda")):
            widget(n, (a, b)[i % 2])
    with right:
        show(plot_schedule(current_config()), cap="Inspection schedule: each tick is one measurement.")

    st.markdown("#### Sampler (aBUS with Subset Simulation)")
    with noted([1], "abus", sec="1")[0]:
        cols = st.columns(4)
        for c, n in zip(cols, ("n_samples_pm", "n_samples_srg", "p0")):
            widget(n, c)

    st.markdown("#### Surrogate and active learning")
    conv = st.session_state["w_convergence_method"]
    tol = {"min_u": "min_u_threshold", "seeded_delta": "seeded_delta_eps"}.get(conv, "convergence_criteria")
    with noted([1], "surrogate", "al", "stopping", sec="1")[0]:
        st.caption("Used only by the surrogate (SRG) methods.")
        cols = st.columns(4)
        for i, n in enumerate(("acquisition_function", "convergence_method", tol, "n_iterations",
                               "min_train_points", "epochs", "lr")):
            widget(n, cols[i % 4])


# ---------------------------------------------------------------- 2 measurements
def plot_measurements(cfg, meas):
    disp, det = meas
    truth = load.true_curves(cfg)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1)
    fig.add_trace(go.Scatter(x=truth.t, y=truth.disp, name="true displacement", line=dict(color="black")), 1, 1)
    fig.add_trace(go.Scatter(x=disp["time"], y=disp["mean"], mode="markers", name="measured displacement",
                             error_y=dict(array=disp["sigma"], color="#0072B2"),
                             marker=dict(color="#0072B2", size=6)), 1, 1)
    fig.add_hline(y=cfg.true_lambda, line_color="black", row=1, col=2, exclude_empty_subplots=False)
    if len(det["time"]):
        fig.add_trace(go.Scatter(x=det["time"], y=det["mean"], mode="markers", name="measured λ",
                                 error_y=dict(array=det["sigma"], color="#D55E00"),
                                 marker=dict(color="#D55E00", size=6, symbol="square")), 1, 2)
    fig.update_xaxes(title_text="t [years]", range=[0, cfg.t_end + 0.5])
    fig.update_yaxes(title_text="displacement [m]", row=1, col=1)
    fig.update_yaxes(title_text="deterioration rate λ [%/year]", row=1, col=2)
    fig.update_layout(height=380, margin=dict(t=20))
    show(fig, cap="Synthetic measurements (markers, ± 1σ) against the true response of the physical twin "
                  "(solid line).")


with st.expander("2  Synthetic measurements"):
    notes_switch("2")
    with noted([1], "measurements", sec="2")[0]:
        st.caption("Drawn from the physical twin and used by the single-data-set and AL-comparison runs, "
                   "while the repeated study draws fresh data for every run.")
        c1, c2, _ = st.columns([1, 1, 3], vertical_alignment="bottom")
        seed = c1.number_input("Measurement seed", value=0, step=1,
                               help="Fixes the measurement noise, so the same seed gives the same data set.")
        if c2.button("Generate measurements"):
            from run_method_comparison import set_seed, generate_measurements
            cfg = current_config()
            set_seed(int(seed))
            st.session_state.meas = generate_measurements(cfg)
            st.session_state.meas_cfg = cfg
        if "meas" in st.session_state:
            if meas_stale():
                st.warning("The beam, true values, schedule or noise changed since these measurements were "
                           "generated. Generate them again before a run.")
            plot_measurements(st.session_state.meas_cfg, st.session_state.meas)

# ---------------------------------------------------------------- run bookkeeping
MODE_NAMES = {"single": "single data set", "repeated": "repeated study", "al": "AL comparison"}


def run_label(r) -> str:
    """'17 Nov 21:35, single data set, 4 methods, done' instead of the folder name."""
    from datetime import datetime
    if r.get("study"):
        c = read_json(os.path.join(r["dir"], "config.json"), {})
        n = r["n_reps"]
        return (f"Study {r['name']}: {n} run{'s' * (n != 1)}, "
                f"σ_δ = {c.get('displacement_sigma', '?')} m")
    job = read_json(os.path.join(r["dir"], "job.json"), {})
    try:
        when = datetime.fromisoformat(r["started"]).strftime("%-d %b %H:%M")
    except (TypeError, ValueError):
        when = r["name"]
    n_al = len(job.get("strategies") or []) * len(job.get("acq_fns") or []) * len(job.get("conv_methods") or [])
    extra = {"single": f"{len(job.get('methods', []))} methods", "repeated": f"{job.get('n_runs')} runs",
             "al": f"{n_al} combinations" if n_al else ""}.get(r["mode"], "")
    return ", ".join(x for x in (when, MODE_NAMES.get(r["mode"], r["mode"]), extra, r["state"]) if x)



def run_steps(job):
    steps = ([METHODS[m] for m in job.get("methods", [])] if job["mode"] == "single" else
             [f"run {i + 1}/{job['n_runs']}" for i in range(int(job["n_runs"]))] if job["mode"] == "repeated"
             else ["AL comparison"])
    return steps + (["reference posterior"] if job.get("reference") and job["mode"] != "al" else [])



# ---------------------------------------------------------------- 3 run
SEED_HELP = ("Fixes the random numbers of the samplers and the surrogate, so a rerun gives the same result. "
             "The data come from the measurement seed in section 2.")

with st.expander("3  Run"):
    notes_switch("3")
    with noted([1], "updating", sec="3")[0]:
        mode = st.radio("Mode", ["single", "repeated", "al"], horizontal=True, format_func={
            "single": "Single data set", "repeated": "Repeated study", "al": "AL comparison"}.get,
            help="Single data set: the four methods once. Repeated study: a new data set per run. "
                 "AL comparison: one data set through every chosen strategy and stopping criterion.\n\n"
                 + docs.topic_tip("workflow"))
        job = {"mode": mode}
        if mode == "single":
            cols = st.columns(4)
            job["methods"] = [k for (k, v), c in zip(METHODS.items(), cols) if c.checkbox(v, value=True, key=f"m_{k}")]
            c1, _ = st.columns([1, 4])
            job["seed"] = c1.number_input("Run seed", value=0, step=1, help=SEED_HELP)
            label = f"Run {len(job['methods'])} method{'s' * (len(job['methods']) != 1)}"
        elif mode == "repeated":
            c1, c2, _ = st.columns([1, 1, 3])
            job["n_runs"] = c1.number_input("Runs", value=10, min_value=1, step=1)
            job["seed"] = c2.number_input("Base seed", value=1000, step=1,
                                          help="Run i uses seed base + i for its data and its samplers.")
            label = f"Start {job['n_runs']} run{'s' * (job['n_runs'] != 1)}"
        else:
            import run_active_learning_comparison as rac
            job["strategies"] = st.multiselect("Strategies", ["sequential", "batch"], default=rac.STRATEGIES,
                                               format_func=load.STRATEGY_NAMES.get)
            job["acq_fns"] = st.multiselect("Learning functions", ACQ, default=rac.ACQ_FNS,
                                            format_func=lambda x: load.ACQ_NAMES.get(x, x))
            job["conv_methods"] = st.multiselect("Stopping criteria", CONV, default=rac.CONV_METHODS,
                                                 format_func=lambda x: load.CONV_NAMES.get(x, x),
                                                 help=CONV_HELP)
            c1, _ = st.columns([1, 4])
            job["max_workers"] = c1.number_input("Parallel workers", value=min(4, os.cpu_count() or 1), min_value=1)
            job["seed"] = 0
            n = len(job["strategies"]) * len(job["acq_fns"]) * len(job["conv_methods"])
            label = f"Run {n} combination{'s' * (n != 1)}"
        if mode != "al":
            job["reference"] = st.checkbox(
                "Compute the reference posterior afterwards", value=False,
                help="Adds time at the end of the run: grid integration plus 200 000 Monte Carlo samples. "
                     "Needed for the accuracy table in section 4.")

        needs_meas = mode in ("single", "al")
        busy = running_run()
        problem = ("A run is already in progress. Wait for it to finish or stop it below." if busy else
                   "Generate measurements first (section 2)." if needs_meas and "meas" not in st.session_state else
                   "The measurements are out of date. Generate them again (section 2)." if needs_meas and meas_stale() else
                   "Select at least one method." if mode == "single" and not job["methods"] else
                   "Select at least one strategy, acquisition function and convergence method." if mode == "al" and not
                   (job["strategies"] and job["acq_fns"] and job["conv_methods"]) else None)
        if problem:
            st.info(problem)
        if st.button(label, type="primary", disabled=bool(problem)):
            run_dir = create_run(current_config(), job, measurements=st.session_state.meas if needs_meas else None)
            start_run(run_dir)
            st.session_state.selected_run = os.path.basename(run_dir)
            st.rerun()

        @st.fragment(run_every=3)
        def live_status():
            r = running_run()
            if not r:
                st.write("No run in progress.")
                return
            status = read_json(os.path.join(r["dir"], "status.json"), {})
            steps, step = run_steps(read_json(os.path.join(r["dir"], "job.json"), {})), status.get("step") or "starting"
            i = steps.index(step) if step in steps else 0
            st.progress(i / len(steps), text=f"{run_label(r)}: step {i + 1} of {len(steps)}, {step}")
            log = os.path.join(r["dir"], "run.log")
            if os.path.exists(log):
                st.code("".join(open(log, errors="replace").readlines()[-15:]) or " ")
            confirm = lambda v: st.session_state.update(confirm_stop=v)
            if not st.session_state.get("confirm_stop"):
                st.button("Stop run…", on_click=confirm, args=(True,))
            else:
                st.warning("Stop this run? Steps that have finished stay saved, but the current step is lost.")
                c1, c2, _ = st.columns([1, 1, 4])
                c1.button("Stop run", type="primary", on_click=lambda: (stop_run(r["dir"]), confirm(False)))
                c2.button("Keep running", on_click=confirm, args=(False,))

        live_status()

# ---------------------------------------------------------------- 4 results
def in_gpa(df):
    df = df.copy()
    df[["E_mu", "E_sigma"]] = df[["E_mu", "E_sigma"]] / 1e6
    return df


@st.cache_data(show_spinner="Drawing from the reference posterior for every time step…")
def reference_draws(run_dir, rep=0):
    meas = load.load_measurements(run_dir, rep)
    if not meas:
        return None
    cfg = CaseStudyConfig.from_json(os.path.join(run_dir, "config.json"))
    return load.reference_samples(cfg, meas)






SLIDER_HELP = ("Selects a time step: the posterior-over-time figures show the updating up to it, the "
               "posterior-sample panels (E against λ) the posterior at it, and, for a single run, the training "
               "panels the points collected up to it.")


def posterior_grid(series, ref, cfg, meas, truth, t=None, what="method", legend=None, split_after=None):
    show(posterior_grid_fig(series, ref, cfg, meas, truth, t, what, legend, split_after),
         cap=posterior_caption(what, t))


def samples_grid(items, ref, cfg, t, what="method", split_after=None):
    """Posterior samples of E and λ at time t, one panel per method on shared axes, each over the
    reference posterior at t (grey). items = [(title, DataFrame with E [kN/m²] and lambda, colour)]."""
    items = [it for it in items if it[1] is not None and len(it[1])]
    if not items:
        return
    ncol = min(len(items), 4) if len(items) != 6 else 3
    nrow = -(-len(items) // ncol)
    fig = make_subplots(rows=nrow, cols=ncol, shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.02, vertical_spacing=0.1, subplot_titles=[it[0] for it in items])
    r_t = ref[ref.t == t] if ref is not None else None
    for i, (title, s, col_) in enumerate(items):
        row, c = i // ncol + 1, i % ncol + 1
        if r_t is not None and len(r_t):
            fig.add_trace(go.Scatter(x=r_t.E / 1e6, y=r_t["lambda"], mode="markers", name="reference posterior",
                                     legendgroup="ref", showlegend=i == 0, hoverinfo="skip",
                                     marker=dict(color="#c8c8c8", size=4)), row, c)
        fig.add_trace(go.Scatter(x=s.E / 1e6, y=s["lambda"], mode="markers", name=title, showlegend=False,
                                 marker=dict(color=col_, size=4, opacity=0.7)), row, c)
        fig.add_trace(go.Scatter(x=[cfg.true_E / 1e6], y=[cfg.true_lambda], mode="markers", name="true value",
                                 legendgroup="true", showlegend=i == 0,
                                 marker=dict(symbol="star", size=12, color="black")), row, c)
    fig.update_xaxes(title_text="E [GPa]", row=nrow)
    fig.update_yaxes(title_text="λ [%/year]", col=1)
    if split_after and nrow == 1 and split_after < len(items):
        split_line(fig, split_after)
    fig.update_layout(height=330 * nrow + 90, legend=dict(orientation="h", yanchor="top", y=-0.2, x=0))
    show(fig, cap=f"Posterior samples of E and λ at t = {t} years, one panel per {what} (colour), over draws from "
                  "the numerical reference posterior at the same time step (grey). The star is the true value.")


def training_grid(items, cloud, what="method", t=None):
    """Surrogate training points in the surrogate's input space (E, I), one panel per method on
    shared axes, over the reference posterior mapped to (E, I) (grey).
    items = [(title, training inputs X [E kN/m², I m⁴] or None, size of the initial design, colour)]."""
    items = [it for it in items if it[1] is not None and len(it[1])]
    if not items:
        return
    ncol = min(len(items), 4) if len(items) != 6 else 3
    nrow = -(-len(items) // ncol)
    fig = make_subplots(rows=nrow, cols=ncol, shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.02, vertical_spacing=0.1, subplot_titles=[it[0] for it in items])
    for i, (title, X, n0, col_) in enumerate(items):
        row, c = i // ncol + 1, i % ncol + 1
        X = np.asarray(X, dtype=float)
        if cloud is not None:
            fig.add_trace(go.Scatter(x=cloud.E / 1e6, y=cloud.I * 1e4, mode="markers", name="reference posterior",
                                     legendgroup="ref", showlegend=i == 0, hoverinfo="skip",
                                     marker=dict(color="#d4d4d4", size=3)), row, c)
        fig.add_trace(go.Scatter(x=X[:n0, 0] / 1e6, y=X[:n0, 1] * 1e4, mode="markers", name="initial design",
                                 legendgroup="init", showlegend=i == 0,
                                 marker=dict(color="black", symbol="circle-open", size=8)), row, c)
        fig.add_trace(go.Scatter(x=X[n0:, 0] / 1e6, y=X[n0:, 1] * 1e4, mode="markers",
                                 name=f"added by active learning ({len(X) - n0})", showlegend=False,
                                 marker=dict(color=col_, size=6, line=dict(width=0.5, color="black"))), row, c)
        fig.add_annotation(text=f"{len(X) - n0} added", xref=f"x{i + 1 if i else ''} domain",
                           yref=f"y{i + 1 if i else ''} domain", x=0.97, y=0.03, xanchor="right",
                           yanchor="bottom", showarrow=False, font=dict(size=12))
    fig.update_xaxes(title_text="E [GPa]", row=nrow)
    fig.update_yaxes(title_text="I [10⁻⁴ m⁴]", col=1)
    fig.update_layout(height=330 * nrow + 90, legend=dict(orientation="h", yanchor="top", y=-0.2, x=0))
    where = st.columns(2)[0] if len(items) == 1 else st     # a single panel would stretch too wide
    show(fig, where, cap=f"Surrogate training points{f' up to t = {t} years' if t is not None else ''}, one panel per "
                  f"{what}: open circles are the initial design, filled circles were added by active learning. "
                  "Grey: the reference posterior mapped to the surrogate inputs (draws of E and I(λ, t) given the "
                  "data up to each time step), the region where the surrogate has to be accurate.")




def block_head(text, colour=None):
    """Large heading that opens a block of results (a strategy, or the cost comparison)."""
    sq = f'<span style="background:{colour}"></span>' if colour else ""
    st.markdown(f'<p class="block-head">{sq}{text}</p>', unsafe_allow_html=True)


@st.cache_data(show_spinner="Scoring the methods against the reference posterior…")
def single_scores(run_dir, rep, methods):
    """{method: accuracy of its final posterior against the numerical reference posterior of this data set}."""
    from reference_posterior import compare_to_reference
    ref, out = reference_final(run_dir, rep), {}
    for m in methods:
        h = load.load_sample_history(run_dir, m, rep)
        if len(h):
            f = h[h.t == h.t.max()]
            out[m] = compare_to_reference(f.E.to_numpy(), f["lambda"].to_numpy(), f.disp.to_numpy(), ref, grid_ref=ref)
    return out


def results_single(run_dir, cfg, states, rep=0):
    """One data set with the four methods: a single run, or run `rep` (0-based) of a study."""
    states = in_gpa(states)
    rep0 = states[states.rep == rep]
    t_values = sorted(rep0.t.unique())
    t = st.select_slider("Time step t [years]", t_values, value=t_values[-1], help=SLIDER_HELP, key=f"t_{rep}")
    meas = load.load_measurements(run_dir, rep)
    truth, draws = load.true_curves(cfg), reference_draws(run_dir, rep)
    cloud = reference_cloud(run_dir, rep) if meas else None
    methods = [m for m in load.METHOD_ORDER if m in set(rep0.method)]

    n_seq = sum(m.startswith("Seq") for m in methods)
    split = n_seq if 0 < n_seq < len(methods) else None     # rule between sequential and batch
    posterior_grid([(m, load.load_sample_history(run_dir, m, rep), rep0[rep0.method == m], COLORS[m])
                    for m in methods], draws, cfg, meas, truth, t, split_after=split)
    items = []
    for m in (m for m in methods if "SRG" in m):
        smp = load.load_samples(run_dir, m, rep, t)
        items.append((m, smp.get("train_X"), smp.get("n_init", 0), COLORS[m]))
    training_grid(items, cloud, t=t)
    samples_grid([(m, pd.DataFrame({k: v for k, v in load.load_samples(run_dir, m, rep, t).items()
                                    if k in ("E", "lambda")}), COLORS[m]) for m in methods], draws, cfg, t,
                 split_after=split)

    block_head("Cost of all methods")
    below = dict(height=520, margin=dict(t=30, b=10, l=70, r=20),
                 legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    scores = single_scores(run_dir, rep, methods) if meas else None
    if scores is not None:       # above both plots, so they line up
        metric = st.radio("Distance to the reference posterior", ["joint_energy", "E_w1", "lambda_w1",
                                                                  "displacement_w1"], horizontal=True,
                          format_func=REF_COLUMNS.get, help=docs.topic_tip("reference"), key=f"metric_{rep}")
    c1, c2 = st.columns(2)
    fig = px.line(rep0, x="t", y="calls_cum", color="method", color_discrete_map=COLORS, markers=True,
                  category_orders={"method": methods}, labels={"t": "t [years]", "calls_cum": "true-model calls"},
                  log_y=True)
    fig.update_layout(**below)
    show(fig, c1, cap="Cumulative number of true-model calls (log scale): every physics-model evaluation up to t, "
                      "including the batch updates of earlier years and the replays of the sequential method "
                      "after a deterioration reading.")
    if scores is not None:
        fin = rep0.loc[rep0.groupby("method").t.idxmax()].set_index("method").calls_cum
        fig = go.Figure([go.Scatter(x=[fin.get(m)], y=[sc[metric]], mode="markers", name=m,
                                    marker=dict(color=COLORS[m], size=11, line=dict(width=0.5, color="black")))
                         for m, sc in scores.items()])
        fig.update_layout(xaxis_title="true-model calls", yaxis_title=REF_COLUMNS[metric], xaxis_type="log", **below)
        fig.update_yaxes(rangemode="tozero")
        show(fig, c2, cap="Cost against accuracy at the final time step (log scale on the calls). Lower left is better.")
    else:
        c2.caption("No measurements stored for this run, so it cannot be compared with the reference posterior.")


@st.cache_data(show_spinner="Computing the reference posterior of every data set (once per study)…",
               persist="disk")
def study_data(run_dir, stamp, _cfg, _states):
    """stamp: newest modification time of the study's files, so a rerun of the study is not served
    from the cache."""
    return load.study_data(run_dir, _cfg, _states)






def study_figure(samples, calls, cfg, n):
    show(study_fig(samples, calls, cfg), cap=study_caption(cfg, n))


def results_repeated(run_dir, cfg, raw_states):
    stamp = max(os.path.getmtime(f) for fs in load._state_files(run_dir).values() for f in fs)
    tab, samples, calls = study_data(run_dir, stamp, cfg, raw_states)
    st.markdown("#### Final posterior over all runs")
    study_figure(samples, calls, cfg, tab.attrs.get("n_data_sets", ""))
    show_tab = pd.DataFrame({
        "E mean [GPa]": tab.E_mu / 1e6, "E std [GPa]": tab.E_sigma / 1e6,
        "λ mean [%/year]": tab.lambda_mu, "λ std [%/year]": tab.lambda_sigma,
        "Displ. mean [m]": tab.disp_mu, "Displ. std [m]": tab.disp_sigma,
        "True-model calls": tab.get("calls_mu"), "Calls std": tab.get("calls_sigma")}, index=tab.index)
    for c in ("True-model calls", "Calls std"):        # the reference has no cost: a dash, not "None"
        show_tab[c] = [("–" if pd.isna(v) else f"{v:,.0f}") for v in show_tab[c]]
    fmt = {c: st.column_config.NumberColumn(format=f) for c, f in (
        ("E mean [GPa]", "%.1f"), ("E std [GPa]", "%.2f"), ("λ mean [%/year]", "%.3f"), ("λ std [%/year]", "%.3f"),
        ("Displ. mean [m]", "%.4f"), ("Displ. std [m]", "%.4f"))}
    st.dataframe(show_tab.rename_axis("Method").reset_index(), hide_index=True, use_container_width=True,
                 column_config=fmt)
    st.markdown(f'<p class="fig-cap"><b>Table.</b> Posterior at the final time step (t = {cfg.t_end}), mean and '
                f'standard deviation pooled over all samples of all {tab.attrs.get("n_data_sets", "")} runs, '
                f'as in Table 2 of the paper. Reference: the numerical reference posterior of each data set (grid integration). '
                f'True values: E = {cfg.true_E / 1e6:.0f} GPa, λ = {cfg.true_lambda:g} %/year. True-model calls: '
                f'cumulative over the whole time frame, mean and standard deviation over the runs.</p>',
                unsafe_allow_html=True)
    if st.checkbox("Show as LaTeX rows (paper units: E in 10⁷ kN/m²)"):
        def num(v, f):
            return "--" if pd.isna(v) else format(v, f)
        lines = [f"{m} & {num(r.E_mu / 1e7, '.2f')} & {num(r.E_sigma / 1e7, '.2f')} & {num(r.lambda_mu, '.3f')} & "
                 f"{num(r.lambda_sigma, '.3f')} & {num(r.disp_mu, '.3f')} & {num(r.disp_sigma, '.3f')} & "
                 f"{num(r.get('calls_mu'), ',.0f').replace(',', '{,}')} & {num(r.get('calls_sigma'), ',.0f').replace(',', '{,}')} \\\\"
                 for m, r in tab.iterrows()]
        st.code("\n".join(lines), language="latex")
    accuracy_table(run_dir, raw_states, repeated=True)

    # ---- one run of the study, shown like a single run; loaded only on request
    n = int(raw_states.rep.max()) + 1
    block_head("Show results for a chosen run")
    key = f"study_rep::{run_dir}"
    with st.form(f"form_{key}", border=False):
        c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
        pick = c1.slider("Run", 1, n, st.session_state.get(key, 1),
                         help="Each run is one data set with its own measurements. Click Show to load it "
                              "(computing its reference posterior takes a few seconds).")
        if c2.form_submit_button("Show", use_container_width=True):
            st.session_state[key] = pick
    if key in st.session_state:
        rep = st.session_state[key] - 1
        st.caption(f"Run {rep + 1} of {n}.")
        results_single(run_dir, cfg, raw_states, rep=rep)


def accuracy_table(run_dir, states, repeated=False):
    """Accuracy table from the run's reference_metrics.csv (written by analyse_method_comparison), with the
    cumulative call counts; for a study one row per method, averaged over the runs."""
    ref = load.load_csv(run_dir, "reference_metrics.csv")
    if ref is None:
        return
    if not states.empty:              # the stored PM counts are per step; use the cumulative cost
        fin = states.loc[states.groupby(["method", "rep"]).t.idxmax()]
        cum = {(m, float(c)): cc for m, c, cc in zip(fin.method, fin.calls, fin.calls_cum)}
        ref["Likelihood Calls"] = [cum.get((m, float(c)), c) if m in ("Seq PM", "Batch PM") else c
                                   for m, c in zip(ref["Method"], ref["Likelihood Calls"])]
    if repeated:
        ref = (ref.drop(columns=["repetition"], errors="ignore").groupby("Method", sort=False)
               .mean(numeric_only=True).reset_index())
    reference_table(ref, title="Accuracy against the reference posterior"
                    + (" (mean over all runs)" if repeated else ""))


@st.cache_data(show_spinner="Computing the reference posterior…")
def reference_final(run_dir, rep=0):
    cfg = CaseStudyConfig.from_json(os.path.join(run_dir, "config.json"))
    return load.reference_at(cfg, load.load_measurements(run_dir, rep), cfg.timeframe[-1], n=200, n_resample=2000)


@st.cache_data(show_spinner="Mapping the reference posterior to the surrogate inputs (E, I)…")
def reference_cloud(run_dir, rep=0):
    cfg = CaseStudyConfig.from_json(os.path.join(run_dir, "config.json"))
    return load.reference_I_cloud(cfg, load.load_measurements(run_dir, rep))


def al_scores(m, ref):
    """Accuracy of each combination's final posterior against the reference."""
    from reference_posterior import compare_to_reference
    rows = []
    for r in m.to_dict("records"):
        fs = r.get("final_samples")
        if isinstance(fs, dict) and len(fs.get("E") or []):
            rows.append({"combo": r["combo"], "strategy": r["strategy"],
                         "criterion": load.CONV_NAMES.get(r["convergence_method"], r["convergence_method"]),
                         "Likelihood Calls": r["nr_likelihood_calls"],
                         **compare_to_reference(fs["E"], fs["lambda"], fs["displacement"], ref, grid_ref=ref)})
    return pd.DataFrame(rows)


STRATEGY_COLORS = {"sequential": COLORS["Seq SRG"], "batch": COLORS["Batch SRG"]}
CRIT_SYMBOLS = dict(zip(load.CONV_NAMES.values(), ["circle", "square", "diamond", "triangle-up",
                                                   "triangle-down", "cross", "star"]))
CRIT_DASH = dict(zip(load.CONV_NAMES.values(), ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot",
                                                "solid"]))


def results_al(run_dir, cfg):
    m = load.load_al_metrics(run_dir)
    if m.empty:
        st.info("No AL-comparison results yet. They appear here as soon as the first combination finishes.")
        return
    ok = m[m.error.isna()].copy()
    order = {k: i for i, k in enumerate(load.CONV_NAMES)}
    ok = ok.assign(_s=ok.strategy.map({"sequential": 0, "batch": 1}), _c=ok.convergence_method.map(order)) \
           .sort_values(["_s", "_c"]).drop(columns=["_s", "_c"])
    post = load.al_posteriors(ok)
    if post.empty:
        st.info("This run was made before the app stored each combination's posterior, so it cannot be "
                "compared with the reference posterior. Run the comparison again to get those figures.")
    else:
        meas, ref = load.load_measurements(run_dir), reference_final(run_dir)
        scores = al_scores(ok, ref)

        st.markdown("#### Posterior over time")
        post = in_gpa(post)
        draws, truth, hist = reference_draws(run_dir), load.true_curves(cfg), load.al_sample_history(ok)
        t_values = sorted(post.t.unique())
        t = st.select_slider("Time step t [years]", t_values, value=t_values[-1], key="al_t", help=SLIDER_HELP)
        combos = [r for r in ok.itertuples() if (post.combo == r.combo).any()]
        crit = lambda r: load.CONV_SHORT.get(r.convergence_method, r.convergence_method)
        cloud = reference_cloud(run_dir)
        for strat in dict.fromkeys(r.strategy for r in combos):
            group = [r for r in combos if r.strategy == strat]
            name = f"{load.STRATEGY_NAMES[strat]} updating"
            block_head(name, STRATEGY_COLORS[strat])
            posterior_grid([(crit(r), hist.get(r.combo), post[post.combo == r.combo], STRATEGY_COLORS[strat])
                            for r in group], draws, cfg, meas, truth, t, what="stopping criterion",
                           legend={crit(r): name for r in group})
            training_grid([(crit(r), r.train_X, int(r.n_init_train or 0), STRATEGY_COLORS[strat]) for r in group],
                          cloud, what="stopping criterion (all 20 years)")
            samples_grid([(crit(r), hist[r.combo][hist[r.combo].t == t] if r.combo in hist else None,
                           STRATEGY_COLORS[strat]) for r in group], draws, cfg, t, what="stopping criterion")

        block_head("Cost of all combinations")
        metric = st.radio("Distance to the reference posterior", ["joint_energy", "E_w1", "lambda_w1",
                                                                  "displacement_w1"], horizontal=True,
                          format_func=REF_COLUMNS.get, help=docs.topic_tip("reference"))
        fig = go.Figure()
        for r in scores.to_dict("records"):
            fig.add_trace(go.Scatter(x=[r["Likelihood Calls"]], y=[r[metric]], mode="markers",
                                     name=f"{load.STRATEGY_NAMES[r['strategy']]}, {r['criterion']}",
                                     marker=dict(color=STRATEGY_COLORS[r["strategy"]], size=11,
                                                 symbol=CRIT_SYMBOLS.get(r["criterion"], "circle"),
                                                 line=dict(width=0.5, color="black"))))
        fig.update_layout(xaxis_title="true-model calls", yaxis_title=REF_COLUMNS[metric], height=520,
                          margin=dict(t=30, b=10, l=70, r=20),
                          legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
        fig.update_yaxes(rangemode="tozero")
        cost_fig = fig
        cost_cap = ("Cost against accuracy: each point is one combination, scored against the numerical reference "
                    "posterior at the final time step. Lower left is better.")
        c1, c2 = st.columns(2)
        fig = go.Figure()
        for r in combos:
            g = post[post.combo == r.combo]
            fig.add_trace(go.Scatter(x=g.t, y=g.calls, mode="lines+markers", name=r.combo,
                                     line=dict(color=STRATEGY_COLORS[r.strategy],
                                               dash=CRIT_DASH.get(load.CONV_NAMES.get(r.convergence_method),
                                                                  "solid"))))
        fig.update_layout(xaxis_title="t [years]", yaxis_title="true-model calls", height=520,
                          margin=dict(t=30, b=10, l=70, r=20),
                          legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
        show(fig, c1, cap="Cumulative number of true-model calls of each combination.")
        show(cost_fig, c2, cap=cost_cap)

        reference_table(scores.drop(columns=["combo", "strategy"]).rename(columns={"criterion": "Method"})
                        .assign(Method=[f"{load.STRATEGY_NAMES[s]}, {c}" for s, c in
                                        zip(scores.strategy, scores.criterion)]),
                        title="Accuracy of each combination against the reference posterior",
                        method_label="Combination")


    rows = [dict(combo=r.combo, strategy=load.STRATEGY_NAMES.get(r.strategy, r.strategy),
                 criterion=load.CONV_NAMES.get(r.convergence_method, r.convergence_method), **h)
            for r in ok.itertuples() for h in (r.accuracy_history or [])]
    if rows:
        st.markdown("#### Surrogate error")
        h = pd.DataFrame(rows)
        metric = st.radio("Error measure", ["nrmse", "nrmse_local"], horizontal=True,
                          format_func={"nrmse": "global (prior region)", "nrmse_local": "local (posterior region)"}.get)
        fig = go.Figure()
        for r in ok.itertuples():
            g = h[h.combo == r.combo]
            if len(g):
                fig.add_trace(go.Scatter(x=g.n_train, y=g[metric], mode="lines+markers", name=r.combo,
                                         line=dict(color=STRATEGY_COLORS[r.strategy],
                                                   dash=CRIT_DASH.get(g.criterion.iloc[0], "solid"))))
        fig.update_layout(xaxis_title="training points", yaxis_title="NRMSE", yaxis_type="log")
        show(fig,
             cap="Surrogate error (NRMSE on a held-out set) as training points are added.")
    if st.checkbox("Show all metrics of this run"):
        st.dataframe(m.drop(columns=["accuracy_history", "posterior", "final_samples", "train_X"], errors="ignore"),
                     use_container_width=True, hide_index=True)


REF_COLUMNS = {"Method": "Method", "repetition": "Data set", "Likelihood Calls": "True-model calls",
               "n_samples": "Samples per level",
               "E_mean_err": "E: mean error", "E_std_ratio": "E: std ratio", "E_w1": "E: W₁",
               "lambda_mean_err": "λ: mean error", "lambda_std_ratio": "λ: std ratio", "lambda_w1": "λ: W₁",
               "displacement_mean_err": "Displ.: mean error", "displacement_std_ratio": "Displ.: std ratio",
               "displacement_w1": "Displ.: W₁", "joint_energy": "Energy distance (E, λ)",
               "mcs_ess": "MCS effective sample size", "log_evidence_err": "log-evidence error"}


def reference_table(ref, title="Accuracy against the reference posterior", method_label="Method"):
    # MCS diagnostics (ESS, log-evidence) stay in run.log; they would overflow the table
    ref = ref.drop(columns=["mcs_ess", "log_evidence_err"], errors="ignore").dropna(axis=1, how="all")
    ref = ref.rename(columns=REF_COLUMNS)
    if "Data set" in ref and ref["Data set"].nunique() == 1:
        ref = ref.drop(columns="Data set")
    ref = ref.rename(columns={"Method": method_label})
    st.markdown(f"#### {title}")
    st.dataframe(ref, use_container_width=True, hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ref.columns
                                if c not in (method_label, "Data set", "True-model calls", "Samples per level")})
    st.markdown('<p class="fig-cap"><b>Table.</b> Each row against the numerical reference posterior at the final '
                'time step. Mean error = (mean − reference mean) / reference std. Std ratio = std / reference std. '
                'W₁ = Wasserstein-1 distance / reference std. The energy distance is computed on the '
                'standardised (E, λ) samples.'
                + (' The Ref MCS rows score a Monte Carlo reference the same way.' if method_label == "Method" else '')
                + '</p>', unsafe_allow_html=True)


with st.expander("4  Results", expanded="selected_run" in st.session_state):
    notes_switch("4")
    with noted([1], "reference", sec="4")[0]:
        runs = load.list_runs()
        if not runs:
            st.info("No runs yet. Start one in section 3 and its results appear here while it runs.")
        else:
            by_name = {r["name"]: r for r in runs}
            names = list(by_name)
            sel = st.session_state.get("selected_run")
            pick = st.selectbox("Run", names, index=names.index(sel) if sel in names else 0,
                                format_func=lambda n: run_label(by_name[n]))
            run = by_name[pick]
            # one placeholder for everything that depends on the run: replaced as soon as another run
            # is picked, so no figures of the previous run linger while the new one is computed
            body = st.empty()
            body.empty()                        # drop the previous run's figures at once
            with body.container():
                status = read_json(os.path.join(run["dir"], "status.json"), {})
                if status.get("error"):
                    st.error(f"This run failed: {status['error']}. The full message is at the end of "
                             f"{os.path.join(run['dir'], 'run.log')}.")
                cfg = CaseStudyConfig.from_json(os.path.join(run["dir"], "config.json"))
                if st.checkbox("Show the configuration of this run"):
                    st.json(cfg.to_dict())
                if run["mode"] == "al":
                    results_al(run["dir"], cfg)
                else:
                    states = load.load_states(run["dir"])
                    if not states.empty:
                        states = load.add_cumulative_calls(states, cfg)
                    if states.empty:
                        st.info("No results written yet. They appear after the first method finishes its first time step.")
                    elif run["mode"] == "single":
                        results_single(run["dir"], cfg, states)
                        accuracy_table(run["dir"], states)
                    else:
                        results_repeated(run["dir"], cfg, states)


# ---------------------------------------------------------------- 5 conclusion
with st.expander("5  Conclusion"):
    st.markdown("Yours to draw.")

# ---------------------------------------------------------------- references
with st.expander("References"):
    render_topic("references")
