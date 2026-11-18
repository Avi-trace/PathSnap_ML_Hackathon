"""Folium prediction maps -> reports/maps/pred_trip_<id>.html (+ _zoom.html) and headless-Chrome screenshots.

Run: python -u scripts/10_folium_prediction_maps.py [--no-tiles]
Metrics in the title box come from a fresh predict_trip + score_prediction run per trip. The reference route layer is for the
OFFLINE report only; the evaluator UI never shows it.
"""
import argparse
import subprocess

import folium
import numpy as np
from _diag_common import *  # noqa
from pathsnap.pipeline import ui_payload

TRIPS = ["00000052", "00000009", "00000040", "00000007", "00000077"]
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
C = dict(gps="#1f5fd1", net="#8a8a8a", ref="#1a9a3a", clf="#ff8c00", brk="#e00000", hmm="#7b2cbf")
ll = lambda pts: [[p[1], p[0]] for p in pts]   # pipeline [lon,lat] -> folium [lat,lon]


def ref_polyline(trip):
    A, B = trip.arcs[:, 0], trip.arcs[:, 1]
    nid = []
    for a in trip.route:
        for n in (A[a], B[a]):
            if not nid or nid[-1] != n:
                nid.append(int(n))
    return [[float(trip.nodes[n, 1]), float(trip.nodes[n, 0])] for n in nid]


def legend_html(tid, m):
    row = lambda col, txt, dash="": f'<div><span style="display:inline-block;width:26px;border-top:{4 if col==C["hmm"] else 3}px {dash or "solid"} {col};vertical-align:middle;margin-right:6px"></span>{txt}</div>'
    return f"""
<div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:rgba(255,255,255,.93);padding:8px 10px;border:1px solid #888;border-radius:4px;font:12px Arial;color:#111;max-width:300px">
<b>Legend</b>
<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{C['gps']};margin-right:6px"></span>Original GPS observations</div>
{row(C['net'], 'Road network near trace (candidate arcs)')}
{row(C['ref'], 'Reference route (OFFLINE report only; never shown in the evaluator UI)')}
{row(C['clf'], 'Classifier-only route', 'dashed')}
<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{C['brk']};margin-right:6px"></span>Classifier route break (arcs not connected)</div>
{row(C['hmm'], 'HMM / Viterbi route (start green, end red marker)')}
</div>
<div style="position:fixed;top:10px;left:50px;z-index:9999;background:rgba(255,255,255,.93);padding:8px 12px;border:1px solid #888;border-radius:4px;font:12px Arial;color:#111">
<b style="font-size:14px">Trip {tid}</b> &nbsp;({m['n_obs']} eligible obs, final tuned bundle)<br>
Pointwise accuracy: classifier <b>{m['pointwise_acc_classifier']:.2%}</b> &rarr; HMM <b>{m['pointwise_acc_hmm']:.2%}</b><br>
Route continuity: classifier {m['route_continuity_classifier']:.2%} &rarr; HMM {m['route_continuity_hmm']:.2%}<br>
Route F1: classifier {m['route_f1_classifier']:.2%} &rarr; HMM {m['route_f1_hmm']:.2%}
</div>"""


def build(trip, payload, m, tiles, bounds, out):
    gps = ll(payload["gps"])
    mp = folium.Map(location=gps[0], zoom_start=14, tiles="OpenStreetMap" if tiles else None, control_scale=True)
    if not tiles:
        mp.get_root().html.add_child(folium.Element("<style>.leaflet-container{background:#f2efe9}</style>"))
    g = folium.FeatureGroup("1 Original GPS observations", show=True)
    for p, t in zip(gps, trip.t):
        folium.CircleMarker(p, radius=2, color=C["gps"], fill=True, fill_opacity=.8, weight=1, tooltip=f"t={t:.0f}s").add_to(g)
    g.add_to(mp)
    g = folium.FeatureGroup("2 Road network (candidate arcs)", show=True)
    for s in payload["network"]:
        folium.PolyLine([[s[1], s[0]], [s[3], s[2]]], color=C["net"], weight=1.5, opacity=.8).add_to(g)
    g.add_to(mp)
    g = folium.FeatureGroup("3 Reference route (offline only)", show=True)
    folium.PolyLine(ref_polyline(trip), color=C["ref"], weight=11, opacity=.55).add_to(g)
    g.add_to(mp)
    g = folium.FeatureGroup("4 Classifier-only route", show=True)
    for s in payload["classifier_only"]["arcs"]:
        folium.PolyLine([[s[1], s[0]], [s[3], s[2]]], color=C["clf"], weight=3, dash_array="6 6").add_to(g)
    for p in ll(payload["classifier_only"]["breaks"]):
        folium.CircleMarker(p, radius=6, color=C["brk"], fill=True, fill_opacity=.5, tooltip="classifier route break").add_to(g)
    g.add_to(mp)
    g = folium.FeatureGroup("5 HMM / Viterbi route", show=True)
    r = ll(payload["hmm"]["route"])
    folium.PolyLine(r, color=C["hmm"], weight=4, opacity=.9).add_to(g)
    if r:
        folium.Marker(r[0], tooltip="HMM route start", icon=folium.Icon(color="green", icon="play")).add_to(g)
        folium.Marker(r[-1], tooltip="HMM route end", icon=folium.Icon(color="red", icon="stop")).add_to(g)
    g.add_to(mp)
    folium.LayerControl(collapsed=False).add_to(mp)
    mp.get_root().html.add_child(folium.Element(legend_html(trip.trip_id, m)))
    mp.fit_bounds(bounds)
    mp.save(out)


def shoot(html, png):
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1400,900",
                    "--virtual-time-budget=15000", f"--screenshot={png}", "file:///" + str(html).replace("\\", "/")],
                   check=True, capture_output=True, timeout=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tiles", action="store_true")
    tiles = not ap.parse_args().no_tiles
    bundle = load_bundle(BUNDLE)
    (ROOT / "reports" / "maps").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "screenshots").mkdir(parents=True, exist_ok=True)
    for tid in TRIPS:
        trip, pred, m, ref, elig = run_trip(bundle, tid)
        payload = ui_payload(pred, trip, m)
        g, c, h = groups(pred, ref, elig)
        full = [[trip.lat.min(), trip.lon.min()], [trip.lat.max(), trip.lon.max()]]
        out = ROOT / "reports" / "maps" / f"pred_trip_{tid}.html"
        build(trip, payload, m, tiles, full, str(out))
        shoot(out, ROOT / "reports" / "screenshots" / f"map_pred_{tid}.png")
        # zoom: longest run of the most interesting disagreement (hmm wrong & clf right for 00000007, clf wrong & hmm right
        # for HMM-helped trips, both wrong otherwise), best 80-obs window
        mask = {"00000007": elig & c & ~h, "00000077": elig & ~c & ~h}.get(tid, elig & ~c & h)
        # window of 80 obs containing the most observations of the chosen disagreement type
        if mask.any():
            w = 80
            cs = np.r_[0, np.cumsum(mask)]
            lo = int(np.argmax(cs[w:] - cs[:-w])) if pred.n_obs > w else 0
            hi = min(lo + w, pred.n_obs - 1)
            bb = [[trip.lat[lo:hi + 1].min() - 3e-4, trip.lon[lo:hi + 1].min() - 3e-4],
                  [trip.lat[lo:hi + 1].max() + 3e-4, trip.lon[lo:hi + 1].max() + 3e-4]]
            zout = ROOT / "reports" / "maps" / f"pred_trip_{tid}_zoom.html"
            build(trip, payload, m, tiles, bb, str(zout))
            shoot(zout, ROOT / "reports" / "screenshots" / f"map_pred_{tid}_zoom.png")
            print(tid, "zoom window obs", lo, hi, "hits", int(mask[lo:hi + 1].sum()))
        print(tid, {k: round(v, 4) for k, v in m.items() if k != "latency_ms"}, g)


main()
