"""HMM diagnostics (facts only) on the 10 student trips -> reports/hmm_diagnostics.csv, hmm_diagnostics_<id>.csv.

Definitions
  eligible obs      aligned reference arc within ELIGIBLE_M (150 m) of the observation (as in pipeline.score_prediction)
  right             predicted arc == aligned reference arc
  range             maximal run of consecutive observation indices satisfying the condition (ineligible obs break a run)
  med_dist_*_m      median distance (m) from the GPS point to the classifier / HMM chosen arc segment, over the range
  hmm_ref_connected_frac  fraction of obs in the range whose HMM arc and reference arc are the same arc or one is reachable
                    from the other in <= 3 directed hops (metrics.transition_flags rule); hmm_ref_connected_all = frac == 1
"""
import numpy as np
import pandas as pd
from _diag_common import *  # noqa

TRIPS = "00000000 00000007 00000009 00000040 00000052 00000056 00000065 00000076 00000077 00000091".split()
DETAIL = ("00000007", "00000077")


def main():
    bundle = load_bundle(BUNDLE)
    summ = []
    for tid in TRIPS:
        trip, pred, m, ref, elig = run_trip(bundle, tid)
        A, B = trip.arcs[:, 0], trip.arcs[:, 1]
        g, c, h = groups(pred, ref, elig)
        summ.append({"trip": tid, "n_obs_total": pred.n_obs, "n_eligible": int(elig.sum()), **g,
                     "pointwise_acc_classifier": m["pointwise_acc_classifier"], "pointwise_acc_hmm": m["pointwise_acc_hmm"]})
        if tid not in DETAIL:
            continue
        dc, dh = seg_dist(pred.G, pred.N, A, B, pred.clf_arc), seg_dist(pred.G, pred.N, A, B, pred.hmm_arc)
        cache, rows = {}, []
        for kind, mask in (("hmm_wrong_clf_right", elig & c & ~h), ("clf_wrong_hmm_right", elig & ~c & h)):
            for s, e in ranges(mask):
                sl = slice(s, e + 1)
                cf = np.mean([connected(int(u), int(v), A, B, cache) for u, v in zip(pred.hmm_arc[sl], ref[sl])])
                rows.append({"trip": tid, "kind": kind, "obs_start": s, "obs_end": e, "n_points": e - s + 1,
                             "t_start_s": float(trip.t[s]), "t_end_s": float(trip.t[e]),
                             "med_dist_clf_arc_m": round(float(np.median(dc[sl])), 2),
                             "med_dist_hmm_arc_m": round(float(np.median(dh[sl])), 2),
                             "hmm_ref_connected_frac": round(float(cf), 3), "hmm_ref_connected_all": bool(cf == 1.0)})
        pd.DataFrame(rows).to_csv(ROOT / "reports" / f"hmm_diagnostics_{tid}.csv", index=False)
        print(tid, len(rows), "ranges")
    df = pd.DataFrame(summ)
    df.to_csv(ROOT / "reports" / "hmm_diagnostics.csv", index=False)
    print(df.to_string())


main()
