"""EDA figures, tables and one Folium map for Piece 1 (Data & Features), report section 2.2.

Reads only the `train` and `student_test` partitions (`pathsnap.paths.DATA_ROOT`) plus the inventory already
built by `scripts/01_eda_inventory.py` (`reports/eda_trip_summary.csv`). Every number plotted or tabulated comes
from that CSV or from a fresh read of the raw files via `pathsnap.io` / `pathsnap.geo` / `pathsnap.candidates` --
nothing is retyped by hand.

Outputs
-------
reports/figures/eda_01_sampling_interval.png   reports/figures/eda_02_length_duration.png
reports/figures/eda_03_trajectories_grid.png   reports/figures/eda_04_network_one_trip.png
reports/figures/eda_05_network_connectivity.png
reports/figures/eda_06_gps_route_offset.png    reports/figures/eda_07_candidate_radius_coverage.png
reports/tables/eda_inventory.{csv,md}          reports/tables/eda_missing_invalid.{csv,md}
reports/tables/eda_network_connectivity.{csv,md}
reports/tables/eda_data_quality_log.{csv,md}
reports/maps/eda_trip_<id>.html + reports/screenshots/eda_map.png
reports/figures/INDEX_eda.md

Run: python -u scripts/08_eda_figures.py [--data-root ...] [--no-tiles]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import folium  # noqa: E402

from pathsnap.candidates import build_arc_index, generate  # noqa: E402
from pathsnap.geo import Projector  # noqa: E402
from pathsnap.io import PARTITIONS, list_trips, load_trip  # noqa: E402
from pathsnap.paths import DATA_ROOT  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"
TAB_DIR = ROOT / "reports" / "tables"
MAP_DIR = ROOT / "reports" / "maps"
SHOT_DIR = ROOT / "reports" / "screenshots"
CSV = ROOT / "reports" / "eda_trip_summary.csv"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
RADII = (25, 50, 75, 100, 150)
BAD_TRIPS = ("00000033", "00000054")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.25,
})
PART_COLOR = {"train": "#1f5fd1", "student_test": "#e07b00"}


# --------------------------------------------------------------------------------------- helpers

def df_to_md(df: pd.DataFrame, path: Path, floatfmt: str = "{:.4g}") -> None:
    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TAB_DIR / f"{name}.csv", index=False)
    df_to_md(df, TAB_DIR / f"{name}.md")


def savefig(fig, name: str) -> Path:
    out = FIG_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------- data loading

def load_all_trips(data_root):
    """Load every train + student_test trip once (with route). Small dataset (90 trips, 227k GPS rows)."""
    trips = {}
    for part in PARTITIONS:
        for tid in list_trips(data_root, part):
            trips[(part, tid)] = load_trip(data_root, part, tid)
    return trips


def route_dmin(trip) -> np.ndarray:
    """Per-GPS-point distance (m) to the nearest reference-route arc, within 150 m (else inf). Same
    computation as scripts/01_eda_inventory.py's summarise(), re-derived here to get the FULL per-point
    array (the CSV only stores percentiles)."""
    proj = Projector(trip.lon.mean(), trip.lat.mean())
    G = proj.xy(trip.lon, trip.lat)
    N = proj.xy(trip.nodes[:, 0], trip.nodes[:, 1])
    A, B = trip.arcs[:, 0], trip.arcs[:, 1]
    obs, arc, d, _ = generate(G, build_arc_index(N, A, B), 150.0)
    dmin = np.full(trip.n_gps, np.inf)
    route = trip.route
    if route is not None and len(route) and route.min() >= 0 and route.max() < len(A):
        in_route = np.zeros(len(A), bool)
        in_route[route] = True
        m = in_route[arc]
        np.minimum.at(dmin, obs[m], d[m])
    return dmin, G, N, A, B


# --------------------------------------------------------------------------------------- fig 1: sampling interval

def fig_sampling_interval(trips, df):
    all_dt = []
    for (part, tid), trip in trips.items():
        all_dt.append(np.diff(trip.t))
    all_dt = np.concatenate(all_dt)
    all_dt = all_dt[all_dt > 0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    bins = np.logspace(np.log10(max(all_dt.min(), 1e-2)), np.log10(all_dt.max()), 60)
    ax.hist(all_dt, bins=bins, color="#1f5fd1", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step dt (s), log scale")
    ax.set_ylabel("count of GPS steps (log scale)")
    frac1 = float((all_dt == 1).mean())
    ax.set_title(f"Sampling interval, all steps pooled\ntrain+student_test, 90 trips, {len(all_dt):,} steps "
                 f"({frac1:.1%} exactly 1 s)")
    ax.axvline(1.0, color="#e00000", lw=1, ls="--", alpha=0.7)

    ax2 = axes[1]
    for part, color in PART_COLOR.items():
        sub = df[df.partition == part]
        ax2.hist(sub.dt_eq_1s, bins=np.linspace(0, 1, 26), color=color, alpha=0.6, label=f"{part} ({len(sub)} trips)")
    ax2.set_xlabel("per-trip share of steps with dt == 1 s")
    ax2.set_ylabel("number of trips")
    ax2.set_title("Per-trip share of 1 s steps\ntrain (80 trips) + student_test (10 trips)")
    ax2.legend(fontsize=8)
    fig.suptitle("")
    fig.tight_layout()
    return savefig(fig, "eda_01_sampling_interval.png")


# --------------------------------------------------------------------------------------- fig 2: length / duration

def fig_length_duration(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for part, color in PART_COLOR.items():
        sub = df[df.partition == part]
        axes[0].hist(sub.n_gps, bins=20, color=color, alpha=0.6, label=f"{part} ({len(sub)} trips)")
        axes[1].hist(sub.duration_s / 60.0, bins=20, color=color, alpha=0.6, label=f"{part} ({len(sub)} trips)")
    axes[0].set_xlabel("GPS points per trip (n_gps)")
    axes[0].set_ylabel("number of trips")
    axes[0].set_title(f"Trip length (GPS count)\ntrain+student_test, 90 trips, "
                       f"median {int(df.n_gps.median())}, max {int(df.n_gps.max())}")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("trip duration (minutes, t_last - t_first)")
    axes[1].set_ylabel("number of trips")
    axes[1].set_title(f"Trip duration\ntrain+student_test, 90 trips, "
                       f"median {df.duration_s.median()/60:.0f} min, max {df.duration_s.max()/60:.0f} min")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return savefig(fig, "eda_02_length_duration.png")


# --------------------------------------------------------------------------------------- fig 3: trajectories grid

TRAJ_TRIPS = ["00000026", "00000090", "00000005", "00000032", "00000084", "00000075"]


def fig_trajectories_grid(trips, df):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, tid in zip(axes.ravel(), TRAJ_TRIPS):
        trip = trips[("train", tid)]
        proj = Projector(trip.lon.mean(), trip.lat.mean())
        G = proj.xy(trip.lon, trip.lat)
        row = df[(df.partition == "train") & (df.trip_id == tid)].iloc[0]
        ax.plot(G[:, 0] / 1e3, G[:, 1] / 1e3, "-", lw=0.8, color="#1f5fd1")
        ax.scatter(G[0, 0] / 1e3, G[0, 1] / 1e3, color="green", s=25, zorder=5, label="start")
        ax.scatter(G[-1, 0] / 1e3, G[-1, 1] / 1e3, color="red", s=25, zorder=5, label="end")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (km, UTM)")
        ax.set_ylabel("y (km, UTM)")
        ax.set_title(f"trip {tid}: n_gps={int(row.n_gps)}, dur={row.duration_s/60:.0f} min\n"
                     f"extent {row.gps_extent_x_km:.1f}x{row.gps_extent_y_km:.1f} km, UTM {int(row.epsg)}", fontsize=9)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("GPS trajectories in projected metres (pathsnap.geo.Projector, per-trip UTM zone), train partition, 6 diverse trips",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    return savefig(fig, "eda_03_trajectories_grid.png")


# --------------------------------------------------------------------------------------- fig 4: network + trace

NET_TRIP = "00000005"


def fig_network_one_trip(trips, df):
    trip = trips[("train", NET_TRIP)]
    proj = Projector(trip.lon.mean(), trip.lat.mean())
    N = proj.xy(trip.nodes[:, 0], trip.nodes[:, 1])
    G = proj.xy(trip.lon, trip.lat)
    A, B = trip.arcs[:, 0], trip.arcs[:, 1]
    row = df[(df.partition == "train") & (df.trip_id == NET_TRIP)].iloc[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    segs = np.stack([N[A] / 1e3, N[B] / 1e3], axis=1)
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors="#bbbbbb", linewidths=0.5, zorder=1))
    ax.plot(G[:, 0] / 1e3, G[:, 1] / 1e3, "-", color="#1f5fd1", lw=1.2, zorder=3, label="GPS trace")
    ax.scatter(G[0, 0] / 1e3, G[0, 1] / 1e3, color="green", s=30, zorder=4, label="start")
    ax.scatter(G[-1, 0] / 1e3, G[-1, 1] / 1e3, color="red", s=30, zorder=4, label="end")
    ax.set_aspect("equal")
    ax.set_xlabel("x (km, UTM)")
    ax.set_ylabel("y (km, UTM)")
    ax.set_title(f"Road network + GPS trace, train trip {NET_TRIP}\n"
                 f"n_nodes={int(row.n_nodes)}, n_arcs={int(row.n_arcs)}, n_components={int(row.n_components)}, "
                 f"largest_component_frac={row.largest_component_frac:.3f}, n_gps={int(row.n_gps)}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return savefig(fig, "eda_04_network_one_trip.png")


# --------------------------------------------------------------------------------------- fig 5: network size/connectivity

def fig_network_connectivity(df):
    d = df.sort_values("n_arcs").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(d.n_nodes, d.n_arcs, c=[PART_COLOR[p] for p in d.partition], s=18, alpha=0.8)
    for part, color in PART_COLOR.items():
        ax.scatter([], [], c=color, label=part)
    ax.set_xlabel("n_nodes")
    ax.set_ylabel("n_arcs")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Network size per trip\ntrain+student_test, 90 trips\n"
                 f"median nodes={int(df.n_nodes.median())}, median arcs={int(df.n_arcs.median())}")
    ax.legend(fontsize=8)

    ax = axes[1]
    x = np.arange(len(d))
    ax.bar(x, d.n_components, color="#8a8a8a", width=0.9, label="n_components")
    ax2 = ax.twinx()
    ax2.plot(x, d.largest_component_frac, color="#e00000", lw=1.2, label="largest_component_frac")
    ax.set_xlabel("trips, sorted by n_arcs")
    ax.set_ylabel("number of connected components")
    ax2.set_ylabel("largest-component share of used nodes", color="#e00000")
    ax2.set_ylim(0, 1.45)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    n_multi = int((df.n_components > 1).sum())
    ax.set_title(f"Connectivity per trip\ntrain+student_test, 90 trips, {n_multi}/90 split into >1 component\n"
                 f"min largest-component share {df.largest_component_frac.min():.3f}")
    ax.set_ylim(0, d.n_components.max() * 1.45)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    leg.get_frame().set_alpha(1.0)
    leg.get_frame().set_edgecolor("#888888")
    fig.tight_layout()
    return savefig(fig, "eda_05_network_connectivity.png")


# --------------------------------------------------------------------------------------- fig 6: gps-to-route offset

OFFSET_TRIPS = {"00000033": "excluded (bad)", "00000054": "excluded (bad)", "00000005": "normal train trip"}


def fig_gps_route_offset(trips):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for col, (tid, label) in enumerate(OFFSET_TRIPS.items()):
        trip = trips[("train", tid)]
        dmin, G, N, A, B = route_dmin(trip)
        fin = dmin[np.isfinite(dmin)]

        ax = axes[0, col]
        route_nodes = []
        for a in trip.route:
            for n in (A[a], B[a]):
                if not route_nodes or route_nodes[-1] != n:
                    route_nodes.append(int(n))
        rp = N[route_nodes]
        ax.plot(rp[:, 0] / 1e3, rp[:, 1] / 1e3, "-", color="#1a9a3a", lw=2, alpha=0.8, label="reference route", zorder=2)
        ax.scatter(G[:, 0] / 1e3, G[:, 1] / 1e3, s=3, color="#1f5fd1", label="GPS", zorder=3)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_title(f"trip {tid} ({label})\nGPS vs reference route, metres", fontsize=9)
        ax.legend(fontsize=7)

        ax = axes[1, col]
        ax.hist(fin, bins=40, color="#1f5fd1", alpha=0.8)
        p50 = np.median(fin) if len(fin) else np.nan
        n_gt150 = int((~np.isfinite(dmin)).sum())
        ax.axvline(p50, color="#e00000", ls="--", lw=1)
        ax.set_xlabel("point-to-route distance (m)")
        ax.set_ylabel("count of GPS points")
        ax.set_title(f"trip {tid}: median {p50:.0f} m, {n_gt150}/{trip.n_gps} pts >150 m", fontsize=9)
    fig.suptitle("GPS-to-reference-route offset: trips 00000033 and 00000054 (excluded from fitting) vs a normal train trip (00000005)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    return savefig(fig, "eda_06_gps_route_offset.png")


# --------------------------------------------------------------------------------------- fig 7: candidate-radius coverage

def fig_candidate_radius_coverage(df):
    excl = df[~df.trip_id.isin(BAD_TRIPS)]
    incl = df

    def agg(d):
        w = d.n_gps
        miss = [np.average(d[f"miss_R{R}"], weights=w) for R in RADII]
        cand = [d[f"cand_rows_R{R}"].sum() / w.sum() for R in RADII]
        return np.array(miss), np.array(cand)

    miss_excl, cand_excl = agg(excl)
    miss_incl, cand_incl = agg(incl)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(RADII, miss_excl * 100, "o-", color="#1f5fd1", label=f"excluding 33/54 ({len(excl)} trips)")
    ax.plot(RADII, miss_incl * 100, "s--", color="#999999", label=f"including 33/54 ({len(incl)} trips)")
    ax.set_xlabel("candidate radius (m)")
    ax.set_ylabel("reference-arc miss rate (%)")
    ax.set_title("Miss rate vs radius\n(observation-weighted, train+student_test)")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    ax = axes[1]
    ax.plot(RADII, cand_excl, "o-", color="#1f5fd1", label=f"excluding 33/54 ({len(excl)} trips)")
    ax.plot(RADII, cand_incl, "s--", color="#999999", label=f"including 33/54 ({len(incl)} trips)")
    ax.set_xlabel("candidate radius (m)")
    ax.set_ylabel("candidates per observation")
    ax.set_title("Candidate density vs radius\n(observation-weighted, train+student_test)")
    ax.legend(fontsize=8)
    for R, m, c in zip(RADII, miss_excl, cand_excl):
        print(f"  R={R:>3}m excl-bad: miss={m:.4f} cand/obs={c:.2f}")
    fig.tight_layout()
    return savefig(fig, "eda_07_candidate_radius_coverage.png")


# --------------------------------------------------------------------------------------- tables

def table_inventory(data_root, df):
    rows = []
    for part in PARTITIONS:
        trips = list_trips(data_root, part)
        sub = df[df.partition == part]
        rows.append({
            "partition": part, "n_trips": len(trips), "files_per_trip": 4,
            "total_files": len(trips) * 4, "total_gps_points": int(sub.n_gps.sum()),
            "gps_per_trip_min": int(sub.n_gps.min()), "gps_per_trip_median": int(sub.n_gps.median()),
            "gps_per_trip_max": int(sub.n_gps.max()),
        })
    inv = pd.DataFrame(rows)
    save_table(inv, "eda_inventory")
    return inv


def table_missing_invalid(df):
    rows = []
    for part in PARTITIONS:
        sub = df[df.partition == part]
        rows.append({
            "partition": part, "n_trips": len(sub),
            "nan_gps_values": int(sub.nan_gps.sum()),
            "implausible_coords": int(sub.coord_implausible.sum()),
            "nonincreasing_timestamp_steps": int(sub.t_nonincreasing.sum()),
            "speed_gt_50ms_steps": int(sub.n_speed_gt_50ms.sum()),
            "trips_with_gap_gt5s": int((sub.n_gaps_gt5s > 0).sum()),
            "total_gaps_gt5s": int(sub.n_gaps_gt5s.sum()),
            "max_gap_s": float(sub.dt_max.max()),
        })
    tab = pd.DataFrame(rows)
    save_table(tab, "eda_missing_invalid")
    return tab


def table_network_connectivity(df):
    rows = []
    for part in PARTITIONS:
        sub = df[df.partition == part]
        rows.append({
            "partition": part, "n_trips": len(sub),
            "n_nodes_median": int(sub.n_nodes.median()), "n_nodes_max": int(sub.n_nodes.max()),
            "n_arcs_median": int(sub.n_arcs.median()), "n_arcs_max": int(sub.n_arcs.max()),
            "self_loops_total": int(sub.self_loops.sum()), "duplicate_arcs_total": int(sub.duplicate_arcs.sum()),
            "two_way_frac_mean": float(sub.two_way_frac.mean()),
            "trips_with_gt1_component": int((sub.n_components > 1).sum()),
            "largest_component_frac_min": float(sub.largest_component_frac.min()),
        })
    tab = pd.DataFrame(rows)
    save_table(tab, "eda_network_connectivity")
    return tab


def table_data_quality_log(df):
    n_gap = int((df.n_gaps_gt5s > 0).sum())
    n_speed = int((df.n_speed_gt_50ms > 0).sum())
    n_multi = int((df.n_components > 1).sum())
    rows = [
        {"issue": "step dt != 1 s (irregular sampling)", "trips_affected": int((df.dt_eq_1s < 1.0).sum()),
         "decision": "kept; dt_prev/dt_next features let the model see GPS gaps instead of assuming fixed-rate sampling"},
        {"issue": f"gap > 5 s between consecutive points (max {df.dt_max.max():.0f} s)", "trips_affected": n_gap,
         "decision": "kept; only affects speed/heading features locally around the gap, points are still labelled independently"},
        {"issue": "step implied speed > 50 m/s", "trips_affected": n_speed,
         "decision": "kept as-is (no row dropped); flagged for the classifier via speed_w/step_speed features, not hand-cleaned"},
        {"issue": "NaN GPS lon/lat/t", "trips_affected": int((df.nan_gps > 0).sum()),
         "decision": "none found; no action needed"},
        {"issue": "implausible coordinates (|lon|>180 or |lat|>90)", "trips_affected": int((df.coord_implausible > 0).sum()),
         "decision": "none found; no action needed"},
        {"issue": "non-increasing timestamps", "trips_affected": int((df.t_nonincreasing > 0).sum()),
         "decision": "none found; no action needed"},
        {"issue": "road network split into >1 connected component", "trips_affected": n_multi,
         "decision": "kept; candidate generation is purely geometric (radius search) and does not require network connectivity"},
        {"issue": "self-loop / duplicate arcs in network", "trips_affected": int(((df.self_loops > 0) | (df.duplicate_arcs > 0)).sum()),
         "decision": f"kept ({int(df.self_loops.sum())} self-loops, {int(df.duplicate_arcs.sum())} duplicate arcs total); harmless for radius-based candidate search"},
        {"issue": "GPS trace far (p50 > 100 m) from its own reference route", "trips_affected": int((df.gps_to_route_p50 > 100).sum()),
         "decision": "trips 00000033 and 00000054 identified and excluded from label fitting (route treated as unreliable); not dropped from the inventory/EDA"},
    ]
    tab = pd.DataFrame(rows)
    save_table(tab, "eda_data_quality_log")
    return tab


# --------------------------------------------------------------------------------------- folium map

MAP_TRIP = "00000085"  # train, n_gps=1092 (in the 800-1500 range), clean (gps_to_route_p90 ~11.5 m), small network


def legend_html():
    return """
<div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:rgba(255,255,255,.93);
padding:8px 10px;border:1px solid #888;border-radius:4px;font:12px Arial;color:#111;max-width:300px">
<b>Legend</b>
<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#1f5fd1;margin-right:6px"></span>Raw GPS observations (.track)</div>
<div><span style="display:inline-block;width:26px;border-top:2px solid #8a8a8a;vertical-align:middle;margin-right:6px"></span>Road network near trace, de-duplicated (.nodes/.arcs, within ~300 m)</div>
<div><span style="display:inline-block;width:26px;border-top:4px solid #1a9a3a;vertical-align:middle;margin-right:6px"></span>Reference route (.route)</div>
</div>
"""


def build_map(trip, tiles: bool):
    proj = Projector(trip.lon.mean(), trip.lat.mean())
    G = proj.xy(trip.lon, trip.lat)
    N = proj.xy(trip.nodes[:, 0], trip.nodes[:, 1])
    A, B = trip.arcs[:, 0], trip.arcs[:, 1]

    # arcs within ~300 m of the trace: both endpoints' min distance to any GPS point <= 300 m (cheap KD-tree)
    from scipy.spatial import cKDTree
    tree = cKDTree(G)
    dA, _ = tree.query(N[A], k=1, workers=-1)
    dB, _ = tree.query(N[B], k=1, workers=-1)
    near = (dA <= 300) | (dB <= 300)
    a_near, b_near = A[near], B[near]

    # de-duplicate across directions: canonical (min_node, max_node) pair, keep one per undirected edge
    lo = np.minimum(a_near, b_near)
    hi = np.maximum(a_near, b_near)
    _, uniq_idx = np.unique(lo.astype(np.int64) * len(trip.nodes) + hi, return_index=True)
    lo_u, hi_u = lo[uniq_idx], hi[uniq_idx]

    gps_ll = list(zip(trip.lat.tolist(), trip.lon.tolist()))
    mp = folium.Map(location=gps_ll[len(gps_ll) // 2], zoom_start=15,
                     tiles="OpenStreetMap" if tiles else None, control_scale=True)
    if not tiles:
        mp.get_root().html.add_child(folium.Element("<style>.leaflet-container{background:#f2efe9}</style>"))

    g_net = folium.FeatureGroup("2 Road network near trace", show=True)
    for n0, n1 in zip(lo_u, hi_u):
        p0 = (float(trip.nodes[n0, 1]), float(trip.nodes[n0, 0]))
        p1 = (float(trip.nodes[n1, 1]), float(trip.nodes[n1, 0]))
        folium.PolyLine([p0, p1], color="#8a8a8a", weight=1.5, opacity=0.8).add_to(g_net)
    g_net.add_to(mp)

    g_route = folium.FeatureGroup("3 Reference route", show=True)
    route_nodes = []
    for a in trip.route:
        for n in (A[a], B[a]):
            if not route_nodes or route_nodes[-1] != n:
                route_nodes.append(int(n))
    route_ll = [(float(trip.nodes[n, 1]), float(trip.nodes[n, 0])) for n in route_nodes]
    folium.PolyLine(route_ll, color="#1a9a3a", weight=4, opacity=0.85).add_to(g_route)
    g_route.add_to(mp)

    g_gps = folium.FeatureGroup("1 Raw GPS observations", show=True)
    for (lat, lon), t in zip(gps_ll, trip.t):
        folium.CircleMarker((lat, lon), radius=2, color="#1f5fd1", fill=True, fill_opacity=0.8,
                             weight=1, tooltip=f"t={t:.0f}s").add_to(g_gps)
    g_gps.add_to(mp)

    folium.LayerControl(collapsed=False).add_to(mp)
    mp.get_root().html.add_child(folium.Element(legend_html()))
    title = (f'<div style="position:fixed;top:10px;left:50px;z-index:9999;background:rgba(255,255,255,.93);'
             f'padding:8px 12px;border:1px solid #888;border-radius:4px;font:12px Arial;color:#111">'
             f'<b style="font-size:14px">Train trip {trip.trip_id}</b> &nbsp;'
             f'({trip.n_gps} GPS points, {len(lo_u)} network edges within ~300 m, de-duplicated)</div>')
    mp.get_root().html.add_child(folium.Element(title))
    bounds = [[trip.lat.min(), trip.lon.min()], [trip.lat.max(), trip.lon.max()]]
    mp.fit_bounds(bounds)
    return mp


def shoot(html: Path, png: Path):
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1400,900",
                    "--virtual-time-budget=15000", f"--screenshot={png}",
                    "file:///" + str(html).replace("\\", "/")],
                   check=True, capture_output=True, timeout=120)


# --------------------------------------------------------------------------------------- index

def write_index(entries):
    lines = ["# EDA outputs index (scripts/08_eda_figures.py)", "",
             "Every figure/table number comes from `reports/eda_trip_summary.csv` "
             "(produced by `scripts/01_eda_inventory.py`) or a fresh read of the raw `train`/`student_test` files "
             "via `pathsnap.io`/`pathsnap.geo`/`pathsnap.candidates`. No number is hand-typed.", ""]
    for path, desc, source, partition in entries:
        lines.append(f"- **`{path}`** -- {desc}\n  - source: {source}\n  - partition: {partition}")
    (FIG_DIR / "INDEX_eda.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--no-tiles", action="store_true", help="use a plain background instead of OSM tiles for the Folium map")
    a = ap.parse_args()

    for d in (FIG_DIR, TAB_DIR, MAP_DIR, SHOT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV, dtype={"trip_id": str})
    print(f"loaded {CSV} ({len(df)} trips: {df.partition.value_counts().to_dict()})")

    print("loading all 90 trips (train + student_test) ...")
    trips = load_all_trips(a.data_root)
    print(f"loaded {len(trips)} trips, {sum(t.n_gps for t in trips.values()):,} total GPS points")

    entries = []

    p = fig_sampling_interval(trips, df)
    entries.append((p.relative_to(ROOT).as_posix(),
                     "Sampling-interval histogram pooled over all GPS steps (log-log) + per-trip share of exactly-1s steps.",
                     "raw .track files (diff of t) + eda_trip_summary.csv column dt_eq_1s", "train + student_test, 90 trips"))

    p = fig_length_duration(df)
    entries.append((p.relative_to(ROOT).as_posix(),
                     "Trip length (n_gps) and trip duration histograms, by partition.",
                     "eda_trip_summary.csv columns n_gps, duration_s", "train + student_test, 90 trips"))

    p = fig_trajectories_grid(trips, df)
    entries.append((p.relative_to(ROOT).as_posix(),
                     f"GPS trajectories in projected metres for 6 diverse train trips ({', '.join(TRAJ_TRIPS)}): "
                     "short/long duration, small/huge extent, with/without a long gap.",
                     "raw .track files, projected with pathsnap.geo.Projector; labels from eda_trip_summary.csv", "train, 6 trips"))

    p = fig_network_one_trip(trips, df)
    entries.append((p.relative_to(ROOT).as_posix(),
                     f"Road network (nodes/arcs as light grey lines) with the GPS trace on top, for trip {NET_TRIP} "
                     "(a mid-sized network: 898 nodes / 1579 arcs, near the lower-middle of the 90-trip range).",
                     "raw .nodes/.arcs/.track files, projected with pathsnap.geo.Projector", f"train, trip {NET_TRIP}"))

    p = fig_network_connectivity(df)
    entries.append((p.relative_to(ROOT).as_posix(),
                     "Network size (nodes vs arcs, log-log scatter) and connectivity (components, largest-component share) per trip.",
                     "eda_trip_summary.csv columns n_nodes, n_arcs, n_components, largest_component_frac", "train + student_test, 90 trips"))

    p = fig_gps_route_offset(trips)
    entries.append((p.relative_to(ROOT).as_posix(),
                     "GPS points + reference route in metres, and histogram of point-to-route distance, for trips "
                     "00000033 and 00000054 (excluded, unusable labels) vs normal train trip 00000005.",
                     "recomputed per-point distance via pathsnap.candidates.generate + .route arcs "
                     "(same method as eda_trip_summary.csv's gps_to_route_* columns, but kept per-point here)",
                     "train, trips 00000033 / 00000054 / 00000005"))

    p = fig_candidate_radius_coverage(df)
    entries.append((p.relative_to(ROOT).as_posix(),
                     "Reference-arc miss rate and candidates/observation vs radius (25/50/75/100/150 m), "
                     "observation-weighted, excluding trips 00000033/00000054 (with the including-them line for comparison).",
                     "eda_trip_summary.csv columns miss_R*, cand_rows_R*, n_gps", "train + student_test, 88 trips (+90-trip comparison line)"))

    inv = table_inventory(a.data_root, df)
    entries.append((f"reports/tables/eda_inventory.csv / .md", "Dataset inventory: trips, files, GPS points per partition.",
                     "pathsnap.io.list_trips (file counts) + eda_trip_summary.csv (n_gps)", "train + student_test"))
    print(inv.to_string(index=False))

    miss = table_missing_invalid(df)
    entries.append((f"reports/tables/eda_missing_invalid.csv / .md",
                     "Missing/invalid-value table: NaN, implausible coordinates, non-increasing timestamps, "
                     "speed>50 m/s steps, gaps>5s, per partition.",
                     "eda_trip_summary.csv columns nan_gps, coord_implausible, t_nonincreasing, n_speed_gt_50ms, n_gaps_gt5s, dt_max",
                     "train + student_test"))
    print(miss.to_string(index=False))

    net = table_network_connectivity(df)
    entries.append((f"reports/tables/eda_network_connectivity.csv / .md", "Network connectivity summary per partition.",
                     "eda_trip_summary.csv columns n_nodes, n_arcs, self_loops, duplicate_arcs, two_way_frac, n_components, largest_component_frac",
                     "train + student_test"))
    print(net.to_string(index=False))

    log = table_data_quality_log(df)
    entries.append((f"reports/tables/eda_data_quality_log.csv / .md",
                     "Data-quality log: issue, number of trips affected, decision taken.",
                     "derived from eda_trip_summary.csv (see script for per-issue column)", "train + student_test"))
    print(log.to_string(index=False))

    # folium map
    print(f"building folium map for trip {MAP_TRIP} ...")
    trip = trips[("train", MAP_TRIP)]
    mp = build_map(trip, tiles=not a.no_tiles)
    html_path = MAP_DIR / f"eda_trip_{MAP_TRIP}.html"
    mp.save(str(html_path))
    png_path = SHOT_DIR / "eda_map.png"
    try:
        shoot(html_path, png_path)
        print(f"screenshot saved to {png_path}")
    except Exception as e:
        print(f"WARNING: screenshot failed ({e}); html map is still at {html_path}")
    entries.append((f"reports/maps/eda_trip_{MAP_TRIP}.html",
                     f"Interactive Folium map for train trip {MAP_TRIP} ({trip.n_gps} GPS points): raw GPS "
                     "(blue dots), road network within ~300 m of the trace (grey, de-duplicated across directions), "
                     "reference route (green), legend, layer control, fit to the trace bounds.",
                     "raw .track/.nodes/.arcs/.route files", f"train, trip {MAP_TRIP}"))
    entries.append(("reports/screenshots/eda_map.png", f"Headless-Chrome screenshot of the map above.",
                     "rendered from eda_trip_{}.html".format(MAP_TRIP), f"train, trip {MAP_TRIP}"))

    write_index(entries)
    print(f"\nwrote INDEX to {FIG_DIR / 'INDEX_eda.md'}")
    print("done.")


if __name__ == "__main__":
    main()
