"""Plotly figures shared by the web app (app/app.py) and the paper script (paper_figures.py).
The builders return figures; the app adds captions and shows them, the script writes files."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from app import load

PALETTE = ["#CC79A7", "#0072B2", "#D55E00", "#009E73"]          # Okabe-Ito, colour-blind safe
COLORS = dict(zip(load.METHOD_ORDER, PALETTE))
FONT = '"Times New Roman", Times, serif'
REF_COLOR = "#8c8c8c"
PRIOR_COLOR = "#e4e4e4"
# rows of the posterior figure: column in the sample tables, axis label, scale to the plotted unit
GRID_ROWS = [("E", "E [GPa]", 1e-6), ("lambda", "λ [%/year]", 1.0), ("I", "I [10⁻⁴ m⁴]", 1e4),
             ("disp", "displacement [m]", 1.0)]

_t = go.layout.Template(pio.templates["simple_white"])
_t.layout.font = dict(family=FONT, color="black", size=14)
_t.layout.colorway = PALETTE
_t.layout.shapedefaults = dict(line=dict(width=1), opacity=1)   # simple_white draws hline/vline at width 0
for ax in (_t.layout.xaxis, _t.layout.yaxis):
    ax.update(mirror=True, ticks="inside", showgrid=False, linecolor="black")
pio.templates["academic"] = _t
pio.templates.default = "academic"


def split_line(fig, after_col):
    """Vertical rule between subplot columns `after_col` and `after_col + 1` (e.g. sequential | batch)."""
    ax = lambda k: fig.layout[f"xaxis{k if k > 1 else ''}"].domain
    x = (ax(after_col)[1] + ax(after_col + 1)[0]) / 2
    fig.add_shape(type="line", xref="paper", yref="paper", x0=x, x1=x, y0=-0.02, y1=1.04,
                  line=dict(color="black", width=1.2))


def posterior_grid_fig(series, ref, cfg, meas, truth, t=None, what="method", legend=None, split_after=None):
    """Posterior of E, λ, I and the displacement at every time step, one column per method,
    drawn as split violins (as in the paper): the method on the right half, the numerical
    reference posterior on the left half.
    series = [(column title, samples (t, E, lambda, I, disp) or None, summary (t, *_mu, *_sigma
    in GPa) used when there are no samples, colour)]; ref = reference samples or None.
    legend: {column title: legend name}, for columns that share a colour (default: the title).
    t: only the time steps up to t are drawn (the axes keep the full range, so sliding plays the
    updating forward)."""
    titles = [s[0] for s in series]
    upto = lambda df: df if t is None or df is None else df[df.t <= t]
    n_rows = len(GRID_ROWS)
    fig = make_subplots(rows=n_rows, cols=len(series), shared_xaxes=True, shared_yaxes="rows",
                        vertical_spacing=0.035, horizontal_spacing=0.015, column_titles=titles)
    spans = {r: [] for r in range(1, len(GRID_ROWS) + 1)}     # values after t = 0, for the axis zoom
    violin = dict(points=False, spanmode="hard", scalemode="width", width=0.95, line_width=0.6,
                  meanline=dict(visible=True, width=1), hoverinfo="skip")
    legend_done = set()

    def once(key):
        new = key not in legend_done
        legend_done.add(key)
        return new

    for c, (title, samples, summary, col_) in enumerate(series, 1):
        name = (legend or {}).get(title, title)
        for row, (var, _, k) in enumerate(GRID_ROWS, 1):
            if ref is not None:
                fig.add_trace(go.Violin(x=upto(ref).t, y=upto(ref)[var] * k, side="negative", line_color=REF_COLOR,
                                        fillcolor=REF_COLOR, opacity=0.75, name="reference posterior",
                                        legendgroup="ref", showlegend=once("ref"), **violin), row, c)
                spans[row].append(ref.loc[ref.t > 0, var] * k)
            if samples is not None and len(samples):
                fig.add_trace(go.Violin(x=upto(samples).t, y=upto(samples)[var] * k, side="positive", line_color=col_,
                                        fillcolor=col_, opacity=0.9, name=name, legendgroup=name,
                                        showlegend=once(("m", name)) and row == 1, **violin), row, c)
                spans[row].append(samples.loc[samples.t > 0, var] * k)
            elif summary is not None and f"{var}_mu" in summary:     # older runs: mean ± 2σ only
                su = upto(summary)
                fig.add_trace(go.Scatter(x=su.t + 0.2, y=su[f"{var}_mu"], mode="markers",
                                         marker=dict(color=col_, size=4), name=f"{title}, mean ± 2σ",
                                         error_y=dict(array=2 * su[f"{var}_sigma"], thickness=1, width=0),
                                         legendgroup=title, showlegend=once(("s", title)) and row == 1), row, c)
                sm = summary[summary.t > 0]
                spans[row] += [sm[f"{var}_mu"] - 2 * sm[f"{var}_sigma"], sm[f"{var}_mu"] + 2 * sm[f"{var}_sigma"]]
        # truth: lines for the constants, crosses for the time-dependent quantities
        tr = upto(truth)
        for row, y in ((3, tr.I * 1e4), (4, tr.disp)):
            fig.add_trace(go.Scatter(x=tr.t, y=y, mode="markers", name="true value", legendgroup="true",
                                     showlegend=once("true"), marker=dict(color="black", size=6, symbol="x-thin",
                                                                          line=dict(width=1.5))), row, c)
        if meas:
            for row, d in ((4, meas["displacement"]), (2, meas["deterioration"])):
                keep = np.asarray(d["time"], dtype=float) <= (np.inf if t is None else t)
                d = {k: np.asarray(v)[keep] for k, v in d.items()}
                if len(d["time"]):
                    fig.add_trace(go.Scatter(x=np.asarray(d["time"]) + 0.25, y=d["mean"], mode="markers",
                                             name="measurement ± 1σ", legendgroup="meas", showlegend=once("meas"),
                                             marker=dict(color="black", size=4, symbol="diamond"),
                                             error_y=dict(array=d["sigma"], thickness=0.8, width=0,
                                                          color="rgba(0,0,0,0.35)")),
                                  row, c)
    fig.add_hline(y=cfg.true_E / 1e6, line=dict(dash="dot", color="black", width=1), row=1, col="all")
    fig.add_hline(y=cfg.true_lambda, line=dict(dash="dot", color="black", width=1), row=2, col="all")
    if t is not None:
        fig.add_vline(x=t, line=dict(color="grey", width=1), row="all", col="all")
    for row, (_, label, _) in enumerate(GRID_ROWS, 1):
        fig.update_yaxes(title_text=label, row=row, col=1)
        v = pd.concat(spans[row]) if spans[row] else None
        if v is not None and len(v):                  # zoom past the wide prior at t = 0
            lo, hi = v.quantile(0.005), v.quantile(0.995)
            pad = 0.1 * (hi - lo) or 0.05 * abs(hi)
            fig.update_yaxes(range=[lo - pad, hi + pad], row=row)
    fig.update_xaxes(range=[-0.7, cfg.t_end + 0.7])
    if split_after:
        split_line(fig, split_after)
    fig.update_xaxes(title_text="t [years]", row=n_rows, dtick=5)
    fig.update_layout(height=950, violinmode="overlay", violingap=0, margin=dict(b=110),
                      legend=dict(orientation="h", yanchor="top", y=-0.07, x=0))
    return fig


def posterior_caption(what="method", t=None):
    """Caption of posterior_grid_fig."""
    return (f"Posterior of E, λ, the second moment of area I and the displacement at every time step, one "
                  f"column per {what}. Each violin is split: the right half (colour) is the {what}'s posterior, "
                  "the left half (grey) the numerical reference posterior given the data measured up to that time "
                  "step, with the line inside marking the mean. Dotted lines and crosses mark the true values, "
                  "diamonds the measurements ± 1σ."
                  + (" The figure shows the time steps up to the one selected above (grey vertical line)."
                     if t is not None else "")
                  + " The y-axes are zoomed to t ≥ 1, so the wide prior at t = 0 is cut off. Double-click a "
                    "panel to see all of it.")


def study_fig(samples, calls, cfg, ncols=4):
    """The paper's Figure 7 in one row: prior (light grey) and posterior pooled over all runs
    (colour; reference in grey) of E, λ and the displacement at the final time step, and the
    cumulative true-model calls."""
    whos = ["Reference"] + [m for m in load.METHOD_ORDER if m in set(samples.who)]
    colour = {"Reference": REF_COLOR, **COLORS}
    panels = [("E", "E [GPa]", 1e-6), ("lambda", "λ [%/year]", 1.0), ("disp", "displacement [m]", 1.0)]
    nrows = 4 // ncols
    fig = make_subplots(rows=nrows, cols=ncols, horizontal_spacing=0.06, vertical_spacing=0.18,
                        subplot_titles=[p[1] for p in panels] + ["true-model calls"])
    vio = dict(points=False, spanmode="hard", scalemode="width", width=0.85, hoverinfo="skip", side="both")
    for c, (var, _, k) in enumerate(panels, 1):
        for who in whos:
            for kind in ("prior", "posterior"):
                v = samples[(samples.who == who) & (samples.kind == kind) & (samples["var"] == var)].value * k
                if not len(v):
                    continue
                col = PRIOR_COLOR if kind == "prior" else colour[who]
                fig.add_trace(go.Violin(x=[who] * len(v), y=v, fillcolor=col, line_color=col, opacity=1,
                                        line_width=0.5, name="prior" if kind == "prior" else who,
                                        legendgroup=kind if kind == "prior" else who,
                                        showlegend=(c == 1 and (kind == "posterior" or who == "Reference")),
                                        meanline=dict(visible=kind == "posterior", color="black", width=1), **vio),
                              (c - 1) // ncols + 1, (c - 1) % ncols + 1)
    truth = load.true_curves(cfg)
    for c, y in ((1, cfg.true_E / 1e6), (2, cfg.true_lambda), (3, float(truth.disp.iloc[-1]))):
        fig.add_hline(y=y, line=dict(dash="dash", color="black", width=1), row=(c - 1) // ncols + 1, col=(c - 1) % ncols + 1)
    for who in whos[1:]:
        v = calls[calls.method == who].calls_cum
        fig.add_trace(go.Violin(x=[who] * len(v), y=v, fillcolor=colour[who], line_color=colour[who], opacity=1,
                                line_width=0.5, showlegend=False, name=who,
                                meanline=dict(visible=True, color="black", width=1), **vio), nrows, ncols)
        fig.add_annotation(x=who, y=np.log10(v.max()) + 0.12, text=f"{v.mean():,.0f}", showarrow=False,
                           font=dict(size=12), xref="x4", yref="y4")
    lo, hi = np.log10(max(calls.calls_cum.min(), 1)), np.log10(calls.calls_cum.max())
    fig.add_annotation(x="Reference", y=(lo + hi) / 2, text="n/a",
                       showarrow=False, font=dict(size=12, color="#777"), xref="x4", yref="y4")
    fig.update_yaxes(type="log", row=nrows, col=ncols)
    fig.update_xaxes(categoryorder="array", categoryarray=whos, tickangle=-45)
    fig.update_layout(height=520 if nrows == 1 else 760, violinmode="overlay", violingap=0.1, margin=dict(b=40),
                      legend=dict(orientation="h", yanchor="top", y=-0.32, x=0))
    return fig


def study_caption(cfg, n):
    """Caption of study_fig."""
    return (f"Final time step (t = {cfg.t_end}) over all {n} runs: prior (light grey) and posterior "
                  "(colour, with the numerical reference posterior in grey) of E, λ and the displacement, with all "
                  "samples of all runs pooled and the line inside marking the mean. Dashed: true values. Right: "
                  "cumulative true-model calls over the whole time frame per run (log scale), with the "
                  "mean written above, as in Figure 7 of the paper.")
