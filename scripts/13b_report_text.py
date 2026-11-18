# Narrative and page assembly for scripts/13_build_report.py (executed inside that script; see it for the available variables).
# The text is written for people: short sentences, what we did, what we saw, what it means, and where we are unsure.

CSS = """
@page { size: A4; margin: 16mm 14mm 18mm 14mm; @bottom-center { content: counter(page); font: 8pt 'Segoe UI', Arial, sans-serif; color: #777; } }
html { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 10pt; line-height: 1.45; color: #1b1f23; }
body { margin: 0; }
h1 { font-size: 30pt; margin: 0 0 4px; letter-spacing: .5px; }
h2 { font-size: 14pt; margin: 22px 0 6px; padding-bottom: 3px; border-bottom: 1.5px solid #d0d5da; break-after: avoid; }
h3 { font-size: 11pt; margin: 14px 0 4px; break-after: avoid; }
h4 { font-size: 10pt; margin: 10px 0 2px; break-after: avoid; }
p { margin: 4px 0 7px; } ul, ol { margin: 4px 0 8px 18px; padding: 0; } li { margin: 2px 0; }
a { color: #1a56a0; text-decoration: none; }
figure { margin: 8px 0; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 112mm; object-fit: contain; border: 1px solid #e3e6ea; }
figcaption, .tcap { font-size: 8.4pt; color: #444; margin: 3px 4px 0; text-align: left; }
.tcap { font-weight: 600; margin-top: 8px; break-after: avoid; }
table { border-collapse: collapse; width: 100%; font-size: 8.2pt; margin: 4px 0 9px; break-inside: auto; }
tr { break-inside: avoid; }
th, td { border: 1px solid #cfd4d9; padding: 2.5px 5px; text-align: left; vertical-align: top; }
th { background: #eef1f4; }
td.n, th.n { text-align: right; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.grid2 figure img { max-height: 70mm; }
.title { min-height: 245mm; display: flex; flex-direction: column; justify-content: center; break-after: page; }
.title .uni { font-size: 10pt; color: #555; margin-bottom: 40px; }
.title h2 { border: 0; font-size: 17pt; font-weight: 500; margin: 0 0 24px; }
.title .sub { font-size: 12pt; color: #555; }
.todo { background: #fff0b3; padding: 1px 5px; border-radius: 3px; }
.box { background: #f4f6f8; border-left: 3px solid #e69a1f; padding: 6px 10px; margin: 8px 0; }
pre { font: 7.3pt/1.35 Consolas, 'Courier New', monospace; background: #f5f6f8; border: 1px solid #dde1e5; padding: 6px 8px; white-space: pre-wrap; margin: 6px 0; break-inside: avoid; break-after: avoid; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 9pt; background: #f1f3f5; padding: 0 3px; border-radius: 2px; }
.small { font-size: 8.5pt; color: #555; }
.model { margin-bottom: 6px; }
.model h4 { break-after: avoid; }
.model p { break-after: avoid; }
.model ul { margin: 2px 0 4px 16px; }
table.wide { font-size: 7pt; }
table.wide th, table.wide td { padding: 2px 3px; }
.names td { font-size: 10.5pt; padding: 3px 12px 3px 6px; }
"""

parts: list[str] = []
add = parts.append
rf = cmp_.loc["RandomForest"]
HA = hmm_agg.loc["tuned decoder"]
DA = hmm_agg.loc["default decoder"]
n_models = len(MODELS)


def n(v, d=0):
    return f"{v:,.{d}f}"


# ------------------------------------------------------------------ title page
add(f"""<section class='title'>
<div class='uni'>PES University · Department of Computer Science and Engineering · RR Campus<br>UE24CS352A: Machine Learning · PathSnap hackathon</div>
<h1>PathSnap</h1>
<h2>Distributed map-matching and route reconstruction</h2>
<p class='sub'>Technical report</p>
<p class='sub' style='margin:0 0 4px'>Team members</p>
<table class='names' style='width:auto;margin:0 0 10px'><thead><tr><th>Name</th><th>SRN</th></tr></thead><tbody>
<tr><td>Avyukt Ranjan</td><td>PES1UG24AM060</td></tr>
<tr><td>Avanthika Jitesh</td><td>PES1UG24AM059</td></tr>
<tr><td>Ashwin Anand</td><td>PES1UG24AM057</td></tr>
<tr><td>Aryan Vinju</td><td>PES1UG24AM056</td></tr>
<tr><td>Aryan Arora</td><td>PES1UG24AM055</td></tr>
</tbody></table>
<p><span class='todo'>Team name: fill in</span></p>
<p>GitHub repository (source code, result tables, figures, Folium maps and run instructions): <a href='{REPO_URL}'>{REPO_URL}</a>{(' (code version: commit <code>' + GIT + '</code>)') if GIT else ''}</p>
<p>18 November 2026</p>
</section>""")

# ------------------------------------------------------------------ summary
add(f"""<h2>Summary</h2>
<p>GPS receivers are noisy, so a vehicle driving along one road can appear to jump onto a parallel street or cut across a building.
PathSnap takes such a trace and works out which roads the vehicle really used. It does this in two steps. A <b>classifier</b> looks at every GPS point and scores the road
segments close to it. Then a <b>hidden Markov model (HMM)</b> picks the single most plausible, connected route from those scores. The model and the road networks live on a server laptop behind a small web API;
a browser interface on a second laptop lets an evaluator pick a trip and see the route drawn on a street map.</p>
<div class='box'><b>Headline results</b> (ten student-test trips, which we used for development; the evaluator trips stay hidden until the demonstration).
<ul>
<li>We trained and compared <b>{n_models} models from {n_fam} algorithm families</b> with cross-validation grouped by trip. <b>Random Forest</b> ranked first on the metric we fixed in advance (observation-level Top-1: {f4(rf.cv_top1_mean)} in cross-validation, {f4(rf.st_top1)} on student-test), but the top models are close, so this is a modest margin and not a decisive win.</li>
<li>Bayesian tuning (Optuna) of the Random Forest lifted cross-validation Top-1 from {f4(tuned.loc['untuned','cv_top1'])} to {f4(tuned.loc['tuned','cv_top1'])}; on student-test the tuned and untuned forests are a tie ({f4(tuned.loc['tuned','st_top1'])} vs {f4(tuned.loc['untuned','st_top1'])}).</li>
<li>Choosing each point's best road independently matches the reference road for <b>{pc(HA.pointwise_acc_classifier)}</b> of GPS points and keeps the route connected {pc(HA.route_continuity_classifier)} of the time.
Adding the HMM raises these to <b>{pc(HA.pointwise_acc_hmm)}</b> and <b>{pc(HA.route_continuity_hmm)}</b>. The gain in continuity is the larger one: disconnected jumps almost disappear.</li>
<li>The HMM does not help everywhere: on trip 00000007 it lowers accuracy ({pc(float(ct[ct.trip=='00000007'].pointwise_acc_classifier.iloc[0]))} to {pc(float(ct[ct.trip=='00000007'].pointwise_acc_hmm.iloc[0]))}). We looked into why in Section 5.</li>
</ul></div>
<p>The sections that follow use the structure required in the hackathon's Deliverables document: dataset understanding and EDA (Section 2), model experimentation and comparison (3), Bayesian hyper-parameter optimisation (4),
HMM and Viterbi route reconstruction (5), Folium visualisation (6), API, UI and deployment (7), and academic integrity and AI-use disclosure (8). Limitations, reproducibility and references are in the appendices.</p>""")

# ------------------------------------------------------------------ 1 problem and approach
ARCH = """<svg viewBox='0 0 760 190' xmlns='http://www.w3.org/2000/svg' style='width:100%;font-family:Segoe UI,Arial,sans-serif'>
<defs><marker id='ah' markerWidth='8' markerHeight='8' refX='7' refY='4' orient='auto'><path d='M0,0 L8,4 L0,8 z' fill='#444'/></marker></defs>
<rect x='8' y='24' width='248' height='142' rx='8' fill='#eef3f8' stroke='#9db2c8'/>
<text x='132' y='52' text-anchor='middle' font-size='14' font-weight='700'>Laptop 2: client</text>
<text x='132' y='80' text-anchor='middle' font-size='11.5'>Web interface (browser)</text>
<text x='132' y='100' text-anchor='middle' font-size='11'>trip picker · street map</text>
<text x='132' y='118' text-anchor='middle' font-size='11'>results · request log</text>
<text x='132' y='146' text-anchor='middle' font-size='10' fill='#555'>holds no model and no data</text>
<rect x='504' y='24' width='248' height='142' rx='8' fill='#fdf3e4' stroke='#e0b072'/>
<text x='628' y='52' text-anchor='middle' font-size='14' font-weight='700'>Laptop 1: server</text>
<text x='628' y='80' text-anchor='middle' font-size='11.5'>FastAPI + trained model</text>
<text x='628' y='100' text-anchor='middle' font-size='11'>road networks of the 10 trips</text>
<text x='628' y='118' text-anchor='middle' font-size='11'>HMM / Viterbi decoder</text>
<text x='628' y='146' text-anchor='middle' font-size='10' fill='#555'>reference routes stay here, never sent</text>
<line x1='262' y1='78' x2='498' y2='78' stroke='#444' stroke-width='1.6' marker-end='url(#ah)'/>
<text x='380' y='68' text-anchor='middle' font-size='10.5'>GET /health · GET /trips · POST /predict {trip_id}</text>
<line x1='498' y1='122' x2='262' y2='122' stroke='#444' stroke-width='1.6' marker-end='url(#ah)'/>
<text x='380' y='142' text-anchor='middle' font-size='10.5'>JSON: routes, road geometry, aggregate scores</text>
</svg>"""
STEPS = ["GPS trace<br>(lon, lat, time)", "Candidate roads<br>(arcs within 75 m)", "18 features<br>per candidate", "Classifier<br>P(arc | GPS point)", "HMM / Viterbi<br>connected route", "Route<br>+ scores"]
PIPE = "<svg viewBox='0 0 760 70' xmlns='http://www.w3.org/2000/svg' style='width:100%;font-family:Segoe UI,Arial,sans-serif'><defs><marker id='ah2' markerWidth='7' markerHeight='7' refX='6' refY='3.5' orient='auto'><path d='M0,0 L7,3.5 L0,7 z' fill='#444'/></marker></defs>"
for i, s in enumerate(STEPS):
    x = 6 + i * 126
    a, b = s.split("<br>")
    fill = "#fdf3e4" if i in (3, 4) else "#eef3f8"
    PIPE += f"<rect x='{x}' y='10' width='108' height='50' rx='6' fill='{fill}' stroke='#9aa7b3'/><text x='{x + 54}' y='31' text-anchor='middle' font-size='10.5' font-weight='600'>{a}</text><text x='{x + 54}' y='46' text-anchor='middle' font-size='9.5'>{b}</text>"
    if i < 5:
        PIPE += f"<line x1='{x + 108}' y1='35' x2='{x + 124}' y2='35' stroke='#444' stroke-width='1.4' marker-end='url(#ah2)'/>"
PIPE += "</svg>"

add(f"""<h2>1. Problem and approach</h2>
<p>A trip is a list of GPS readings (about one per second). The road network is a set of nodes (junctions) and directed arcs (road segments, one per direction of travel).
The task is to say, for every reading, which arc the vehicle was on, and to return a route that is physically possible, meaning that consecutive arcs connect.
Reading by reading, noise makes the best-looking arc wrong now and then; a connected route needs a model that looks at the whole sequence.</p>
<p>Our design follows that idea. For each GPS point we collect every arc within 75 m (the <i>candidates</i>) and describe each candidate with 18 numbers: how close it is, which way it points compared with the vehicle's direction of travel, how fast the vehicle moves, and so on.
A classifier turns those numbers into a probability that the candidate is the right one. The HMM then treats the candidates of each point as its possible states, uses the classifier's probability as the evidence for a state,
and uses the road network to decide which moves between states are possible. Viterbi decoding returns the best connected chain.</p>
<figure>{PIPE}<figcaption>Figure 1. The processing pipeline. The two highlighted steps are the machine-learning parts.</figcaption></figure>
<p><b>Data partitions.</b> The dataset has three partitions: 80 <i>train</i> trips (fitting, cross-validation and tuning), 10 <i>student-test</i> trips (development comparisons) and 10 hidden <i>evaluator</i> trips (final demonstration only).
Whole trips are never split across partitions. All numbers in this report come from train and student-test; no code in the repository reads the evaluator partition.</p>
<figure>{ARCH}<figcaption>Figure 2. Two-laptop deployment. The client only asks for a trip by its identifier; everything else happens on the server, which also keeps the reference routes private.</figcaption></figure>""")

# ------------------------------------------------------------------ 2 EDA
inv = T["inventory"]
inv_t = tbl(pd.DataFrame({"Partition": inv.partition, "Trips": inv.n_trips, "Files per trip": inv.files_per_trip, "GPS points": inv.total_gps_points.map(lambda v: n(v)),
                          "Points per trip (min / median / max)": [f"{a} / {b} / {c}" for a, b, c in zip(inv.gps_per_trip_min, inv.gps_per_trip_median, inv.gps_per_trip_max)]}),
            "Table 1. Dataset inventory (partitions train and student_test).")
mi = T["missing_invalid"]
mi_t = tbl(pd.DataFrame({"Partition": mi.partition, "NaN values": mi.nan_gps_values, "Implausible coordinates": mi.implausible_coords, "Non-increasing timestamps": mi.nonincreasing_timestamp_steps,
                         "Steps implying > 50 m/s": mi.speed_gt_50ms_steps, "Trips with a gap > 5 s": mi.trips_with_gap_gt5s, "Largest gap (s)": mi.max_gap_s.map(lambda v: n(v))}),
           "Table 2. Missing and invalid values (partitions train and student_test).")
nw = T["network_connectivity"]
nw_t = tbl(pd.DataFrame({"Partition": nw.partition, "Nodes per trip (median / max)": [f"{n(a)} / {n(b)}" for a, b in zip(nw.n_nodes_median, nw.n_nodes_max)],
                         "Arcs per trip (median / max)": [f"{n(a)} / {n(b)}" for a, b in zip(nw.n_arcs_median, nw.n_arcs_max)],
                         "Self-loops / duplicate arcs": [f"{n(a)} / {n(b)}" for a, b in zip(nw.self_loops_total, nw.duplicate_arcs_total)],
                         "Arcs with a reverse arc": nw.two_way_frac_mean.map(lambda v: pc(v)), "Trips with > 1 component": nw.trips_with_gt1_component,
                         "Smallest share of the biggest component": nw.largest_component_frac_min.map(lambda v: pc(v))}),
            "Table 3. Road-network size and connectivity (partitions train and student_test).")
dq = T["data_quality_log"]
dq_t = tbl(pd.DataFrame({"Issue": dq.issue, "Trips affected": dq.trips_affected, "What we decided": dq.decision}), "Table 4. Data-quality log: what we found and what we did about it.")
f_tr = DSUM["train"]
if FI is not None:
    _top = FI.head(4)
    fi_sentence = ("In the final tuned Random Forest the four features with the highest importance are " + ", ".join(f"{k} ({v:.2f})" for k, v in _top.items())
                   + " (impurity-based importance, which sums to 1 over the 18 features and favours features with many distinct values, so it is a rough guide and not proof).")
else:
    fi_sentence = ""
FEATS = [("dist", "Distance from the GPS point to the arc (m)", "closeness"), ("dist_gap", "Distance minus the nearest candidate's distance", "how much worse than the best alternative"),
         ("dist_rank", "Rank among the point's candidates by distance (0 = nearest)", "closeness, relative"), ("n_cand", "Number of candidates for this point", "local road density"),
         ("along_m, remain_m", "Distance along the arc to the projected point, and what is left of the arc", "where on the segment the point falls"), ("arc_len", "Length of the arc (m)", "short stubs behave differently from long roads"),
         ("lateral", "Signed side offset (left or right of the arc direction)", "separates the two opposite arcs of one road"),
         ("cos_dh, abs_dh", "Heading difference between the GPS direction (from points i-2 to i+2) and the arc direction", "direction of travel; neutral values when the vehicle is not moving"),
         ("cos_gap", "Best cos_dh among the candidates minus this candidate's", "heading, relative"), ("speed_w, step_speed", "Speed over the 5-point window and over the last step (m/s)", "slow points have unreliable heading"),
         ("dt_prev, dt_next", "Seconds to the previous and next GPS reading", "makes GPS gaps visible"), ("outdeg_end, indeg_start", "Number of arcs leaving the arc's end node / entering its start node", "junction structure"),
         ("two_way", "1 if the arc has a reverse arc", "two-way versus one-way road")]
feat_t = tbl(pd.DataFrame(FEATS, columns=["Feature(s)", "Meaning", "Why we use it"]), "Table 6. The 18 features of a candidate (all computed for every candidate of every GPS point).")
add(f"""<h2>2. Dataset Understanding and Exploratory Data Analysis</h2>
<p class='small'>Partitions used here: train (80 trips) and student_test (10 trips) for every analysis; each figure says which. Nothing from the evaluator partition was looked at.</p>
<h3>2.1 What the data is</h3>
<p>Each trip is a folder with four headerless text files: <code>.track</code> (longitude, latitude, time), <code>.nodes</code> (junction coordinates), <code>.arcs</code> (pairs of node ids, one per directed road segment) and <code>.route</code> (the ordered arc ids the vehicle used; the reference).
A GPS row does not correspond to a node or arc row: the only link between them is geometry (which arc is this point near?). Altogether there are {n(n_gps_all)} GPS points.</p>
{inv_t}
<h3>2.2 How did we preprocess the data?</h3>
<ol>
<li><b>Coordinates to metres.</b> The trips come from all over the world ({n_zones} different UTM zones), and degrees are not distances, so every trip is projected into its own UTM zone (metres; scale error at most about 0.1%).</li>
<li><b>Candidate arcs.</b> A spatial index finds every arc within 75 m of each GPS point by exact point-to-segment distance. Long arcs are split into pieces of at most 40 m for the index so that none is missed.</li>
<li><b>Labels.</b> The data gives the route, not "which arc is each point on". We align the points to the route with a dynamic programme that never moves backwards along the route and minimises the distance between each point and its assigned arc (Section 2.5).</li>
<li><b>Features.</b> 18 numbers per candidate (Section 2.6). Tree models need no scaling. Logistic Regression and the neural network standardise their inputs using training data only.</li>
<li><b>Exclusions.</b> Two training trips with unusable reference routes are kept in the EDA but excluded from fitting (Section 2.4).</li>
<li><b>Missing values.</b> There were none to fill (Table 2). GPS rows with a non-finite value would be dropped by the loader; none occurred.</li>
</ol>
<h3>2.3 Missing-value analysis and data quality</h3>
{mi_t}{nw_t}{dq_t}
<h3>2.4 What patterns did we observe during data exploration?</h3>
<p class='small'>GPS trajectory plots: Figures 4 and 5a. Road-network visualisation: Figure 5a. Geographical map made with Folium: Figure 6.</p>
<ul>
<li><b>Sampling is mostly 1 Hz but not always.</b> {pc(dt_pool)} of all steps are exactly one second apart, yet 47 trips contain a gap longer than 5 s (one gap inside trip 00000090 is 153,064 s) and 1,538 steps imply a speed above 50 m/s. We kept these points and gave the classifier the time gaps and speeds as features instead of deleting data.</li>
<li><b>Road networks are fragmented and slightly messy.</b> 89 of the 90 networks consist of several disconnected pieces, and there are duplicate arcs and self-loops. Because candidate search is purely geometric, this does not matter for it. About 92% of arcs have a reverse arc (two-way roads).</li>
<li><b>Reference routes are almost always connected</b> (99.9% of consecutive route arcs link head to tail) and all ids are valid. They are reference data, not perfect truth.</li>
<li><b>Two training trips have unusable labels.</b> For trips 00000033 and 00000054 (12% of train points) the GPS points lie a median of 122 m and 125 m from their own reference route, and the distances have two peaks (Figure 5b): some points are on the route, many are 130 to 140 m away. A single constant shift could not explain two separate peaks. We excluded them from fitting and report this as a limitation of the reference data.</li>
<li><b>A plain "nearest arc" is a poor classifier.</b> Because every two-way road has two arcs at about the same distance, picking the nearest arc is right only {pc(f_tr['nearest_arc_top1'])} of the time. Direction of travel matters, which is why heading features are among our 18.</li>
</ul>
{pair([(R/'figures'/'eda_01_sampling_interval.png', 'Figure 3. Time between consecutive GPS readings (all 90 trips; partitions train and student_test).'), (R/'figures'/'eda_03_trajectories_grid.png', 'Figure 4. Six trajectories in projected metres (partition train).')])}
{pair([(R/'figures'/'eda_04_network_one_trip.png', 'Figure 5a. Road network (grey) and GPS trace of one trip (partition train).'), (R/'figures'/'eda_06_gps_route_offset.png', 'Figure 5b. Trips 00000033 and 00000054 against a normal trip: GPS-to-route distance (partition train).')])}
{fig(R/'screenshots'/'eda_map.png', 'Figure 6. Interactive Folium map of train trip 00000085: raw GPS points (blue), road network near the trace (grey) and the reference route (green), with layer controls. The HTML file is reports/maps/eda_trip_00000085.html.', '88%')}
<h3>2.5 How the labels are made</h3>
<p>For every GPS point we want the reference arc it belongs to. We have the ordered list of arcs the vehicle used, so we assign each point to an arc on that list so that (a) the assignment never moves backwards along the list and (b) the total point-to-arc distance is as small as possible.
This is solved exactly with dynamic programming. The "never backwards" rule is what makes it work on routes that revisit a road or run parallel to themselves, where a plain nearest-arc rule would jump between the two passes.
A point whose assigned arc is more than 150 m away gets no label and is left out of training and scoring.</p>
<h3>2.6 Candidates, and which features did we extract and why?</h3>
<p>We chose the 75 m search radius by measurement (Table 5). A smaller radius misses the true road more often, a larger one multiplies the number of rows. At 75 m the reference arc is missing from the candidate set for {pc(f_tr['miss_rate'], 2)} of training points
({pc(DSUM['student_test']['miss_rate'], 2)} on student-test), with {f_tr['rows_per_obs']:.1f} candidates per point. That gives {n(f_tr['rows'])} training rows from {n(f_tr['eligible_obs'])} labelled GPS points, of which {pc(f_tr['positive_rate'])} are the correct road, so the classes are imbalanced.</p>
{tbl(cover, 'Table 5. Candidate radius: how often the nearest reference-route arc is farther than the radius, and how many candidates result (partitions train and student_test, without the two excluded trips, weighted by GPS points). This counts every GPS point; the percentages quoted in the text count only labelled points, so the two differ slightly.')}
{fig(R/'figures'/'eda_07_candidate_radius_coverage.png', 'Figure 7. Candidate-radius coverage with and without the two excluded trips (partitions train and student_test).', '78%')}
<p>The 18 features answer the questions a person would ask about a candidate road: <i>is it close?</i> (dist, dist_gap, dist_rank, lateral); <i>does it point the way the vehicle is going?</i> (cos_dh, abs_dh, cos_gap);
<i>is the vehicle moving fast enough for its direction to be trusted?</i> (speed_w, step_speed); <i>where on the road is the point?</i> (along_m, remain_m, arc_len); <i>what does the local network look like?</i> (n_cand, outdeg_end, indeg_start, two_way);
and <i>is there a gap in the GPS data?</i> (dt_prev, dt_next). {fi_sentence}</p>
{feat_t}""")

# ------------------------------------------------------------------ 3 models
MT = {
    "GaussianNB": ("A very cheap probabilistic baseline. It shows how far simple per-feature statistics go before any interaction between features is learned.",
                   "It trains in under a second and outputs probabilities. But it assumes the features are independent given the class, which is false here (distance, rank and gap move together), and its probabilities are the poorest of the nine (see the log loss in Table 7).",
                   "Maximum-likelihood fit of one Gaussian per feature and class; there is no separate loss to tune."),
    "LogisticRegression": ("A linear model with well-behaved probabilities. It tells us how much of the problem is linear.",
                           "Fast and stable, and it gives probabilities directly, but one linear boundary cannot express rules such as 'close AND heading the same way AND not a short stub'.",
                           "Log loss (cross-entropy) with L2 regularisation, solved with L-BFGS; inputs standardised with training statistics."),
    "DecisionTree": ("A single tree can learn rule-like interactions directly and can be read by a person.",
                     "Readable and quick, but one tree is unstable and, at depth 12, partly memorises the training trips.",
                     "Splits chosen by Gini impurity; a leaf's class frequencies are its probabilities."),
    "RandomForest": ("Averaging many different trees keeps their non-linear power while cancelling much of the variance of a single tree; the averaged leaf frequencies are usable probabilities.",
                     "Robust, little tuning needed and good probabilities. It is slower to train and bigger than the boosted models, but prediction is fast enough for our use.",
                     "Gini impurity for each tree; 50 trees on bootstrap samples; the forest outputs the average of the trees' class frequencies."),
    "ExtraTrees": ("Like a forest, but split thresholds are drawn at random, which lowers variance and speeds up training.",
                   "The fastest forest to fit. Its probabilities came out less sharp than the Random Forest's (higher log loss).",
                   "Gini impurity with random thresholds."),
    "HistGradientBoosting": ("Boosting builds the model by repeatedly correcting the previous errors; histogram binning makes it fast on millions of rows.",
                             "Strong on tabular data, but with 300 iterations and no early stopping it keeps fitting noise.",
                             "Logistic (binary log) loss, optimised by gradient boosting."),
    "LightGBM": ("Another gradient-boosting implementation, with leaf-wise tree growth; very fast.",
                 "Fast and accurate on many problems. With our default settings its probabilities were the worst of the tree models (high log loss) and its route continuity was the lowest.",
                 "Binary logistic loss, gradient boosting."),
    "XGBoost": ("A widely used gradient-boosting library with second-order optimisation and built-in regularisation.",
                "Accurate and fast. With depth 8 and 300 rounds it fits the training data very closely (largest train-to-CV gap of all models).",
                "Binary logistic loss with second-order (Newton) boosting."),
    "MLP(PyTorch)": ("A small neural network can learn smooth non-linear combinations of the features and, unlike the tree models, trains with an epoch-by-epoch loss we can inspect.",
                     "Flexible and cheap at prediction time. It needs standardised inputs and is sensitive to training settings; it showed no clear advantage over the forest here.",
                     "Binary cross-entropy with the Adam optimiser (learning rate 1e-3, weight decay 1e-5), 15 epochs, batch 4096."),
}


def fit_sentence(m: str) -> str:
    g, ta = tv.loc[m, "train_minus_cv_top1"], tv.loc[m, "train_top1"]
    base = f"Training Top-1 is {f3(ta)} against {f3(tv.loc[m, 'cv_top1_mean'])} in cross-validation (gap {g:+.3f}) and {f3(tv.loc[m, 'st_top1'])} on student-test. "
    if g < 0.012:
        return base + ("The gap is tiny, so there is no sign of over-fitting. " + ("Its scores are also among the lowest, which points to under-fitting: the model is too simple for the task." if tv.loc[m, "cv_top1_mean"] < 0.86 else "It generalises as well as it trains."))
    if g < 0.03:
        return base + "That is a mild over-fit: it does somewhat better on data it has seen, but the cross-validation and student-test scores agree with each other."
    return base + "That is clear over-fitting: the model memorises part of the training trips, and held-out trips score noticeably lower."


rk_cv = cmp_.cv_top1_mean.rank(ascending=False, method="min").astype(int)
rk_st = cmp_.st_top1.rank(ascending=False, method="min").astype(int)
fold_pv = folds.pivot_table(index="model", columns="fold", values="top1")
fold_rank = fold_pv.rank(ascending=False, method="min").astype(int)
behind = float((fold_pv.max() - fold_pv.loc["RandomForest"]).max())          # how far Random Forest is behind the best model in its worst fold
bp = ob["best_params"]

rows31 = []
for m in MODELS:
    fm = folds[folds.model == m]
    rows31.append([m, cmp_.loc[m, "family"], html.escape(cmp_.loc[m, "config"]), f"{fm.macro_f1.mean():.3f} ± {fm.macro_f1.std():.3f}", f4(cmp_.loc[m, "cv_top1_mean"]), f4(cmp_.loc[m, "st_top1"]),
                   f3(cmp_.loc[m, "st_route_continuity"]), f3(cmp_.loc[m, "st_macro_f1"]), f3(cmp_.loc[m, "st_logloss"]), f"{cmp_.loc[m, 'infer_s_per_100k_rows']:.3f} s"])
t31 = tbl(pd.DataFrame(rows31, columns=["Model", "Algorithm family", "Key configuration", "CV Macro F1-score (mean ± std)", "CV Top-1 accuracy", "Student-test Top-1 accuracy",
                                        "Route continuity (student-test)", "Student-test Macro F1-score", "Log loss (student-test)", "Inference time (per 100k rows)"]),
          "Table 7. Consolidated comparison. CV = 3-fold cross-validation grouped by trip on the 78 usable train trips; student-test = the 10 student-test trips after refitting on all usable train trips. Top-1 is observation-level. Route continuity is of the classifier-only route (no HMM). All metrics are applicable to all nine models.",
          cls="wide")
rows31b = []
for m in MODELS:
    r = cmp_.loc[m]
    rows31b.append([m, f3(r.st_precision), f3(r.st_recall), f"{r.fit_s_full:.0f} s", f"{r.cv_fit_s_mean:.0f} s", f"{tv.loc[m, 'train_top1']:.3f}", f"{r.cv_top1_mean:.3f} ± {r.cv_top1_std:.3f}"])
t31b = tbl(pd.DataFrame(rows31b, columns=["Model", "Precision (student-test)", "Recall (student-test)", "Training time, all rows", "Training time per CV fold (30% subsample)", "Train Top-1", "CV Top-1 (mean ± std over folds)"]),
           "Table 7b. Supplementary metrics asked for in the models guideline. Precision and recall are for the correct-arc class at threshold 0.5. Candidate miss rate: N/A per model, because candidates are generated before any model is applied; it is "
           + pc(DSUM['train']['miss_rate'], 2) + " on train and " + pc(DSUM['student_test']['miss_rate'], 2) + " on student-test for every model.")


def pr_sentence(r) -> str:
    p, q = r.st_precision, r.st_recall
    if q - p > 0.05:
        return (f"At threshold 0.5 it finds {q:.0%} of the correct arcs but only {p:.0%} of the arcs it flags are correct: it over-selects, which is what happens when the classes are imbalanced "
                f"(only {pc(f_tr['positive_rate'])} of candidate rows are correct) and the model's probabilities are not calibrated.")
    if p - q > 0.03:
        return f"At threshold 0.5 it is conservative: precision {p:.0%} is higher than recall {q:.0%}, so it misses some correct arcs rather than flagging wrong ones."
    return f"At threshold 0.5 precision ({p:.0%}) and recall ({q:.0%}) are balanced."


per_model = ""
for i, m in enumerate(MODELS, 1):
    a, b, c = MT[m]
    r = cmp_.loc[m]
    fr = [int(fold_rank.loc[m, k]) for k in sorted(fold_rank.columns)]
    per_model += f"""<div class='model'><h4>3.2.{i} {html.escape(m)} <span class='small'>({r.family})</span></h4>
<p><b>A. Model understanding</b></p>
<ul><li><b>Q1. Why was it considered suitable?</b> {a} In our experiments it ranks {rk_cv[m]} of {n_models} on cross-validation Top-1 and {rk_st[m]} of {n_models} on student-test Top-1.</li>
<li><b>Q2. Strengths and limitations for this task.</b> {b}</li></ul>
<p><b>B. Training and configuration</b></p>
<ul><li><b>Q5. Important hyper-parameters and their initial values.</b> {html.escape(r.config)}. Seed 0; all settings are saved in <code>reports/model_configs.json</code>.</li>
<li><b>Q6. Loss function / optimisation objective.</b> {c}</li></ul>
<p><b>C. Experimental results</b></p>
<ul><li><b>Q7. Performance on the student-test dataset.</b> On the ten student-test trips the model (classifier only, before any HMM) picks the reference arc for {pc(r.st_top1)} of GPS points, and the arcs it picks form a connected route {pc(r.st_route_continuity)} of the time. {pr_sentence(r)}
Fitting on all training rows took {r.fit_s_full:.0f} s; scoring 100,000 candidate rows takes {r.infer_s_per_100k_rows:.3f} s.</li>
<li><b>Q8. Observation-level Top-1, Macro F1 and log loss on student-test.</b> Top-1 {f4(r.st_top1)}; Macro F1 {f4(r.st_macro_f1)}; log loss {f4(r.st_logloss)}.</li>
<li><b>Q9. Over-fitting or under-fitting, and the evidence.</b> {fit_sentence(m).strip()} Cross-validation Top-1 in the three folds ranges from {fold_pv.loc[m].min():.3f} to {fold_pv.loc[m].max():.3f} (this model's rank among the nine in each fold: {fr[0]}, {fr[1]}, {fr[2]}).</li></ul></div>"""

best_cv = cmp_.cv_top1_mean.idxmax()
sd = cmp_.cv_top1_std.median()
ll_sorted = cmp_.st_logloss.sort_values()
inf_sorted = cmp_.infer_s_per_100k_rows.sort_values()
rank_ll = int(ll_sorted.index.get_loc("RandomForest")) + 1
pipe_txt = f" The whole server pipeline (candidates, classifier and HMM) needed on average {lat.pipeline_ms.mean() / 1000:.1f} s per student-test trip (Table 13)." if lat is not None else ""
tv_rf = tv.loc["RandomForest"]
sel_rows = [
    ["Generalisation performance", "Top-1 on training rows, in cross-validation and on student-test; the training-minus-CV gap (Figure 11)",
     f"Random Forest: {tv_rf.train_top1:.3f} / {tv_rf.cv_top1_mean:.3f} / {tv_rf.st_top1:.3f}, gap {tv_rf.train_minus_cv_top1:+.3f}. Rank {rk_cv['RandomForest']} of {n_models} on CV Top-1 and {rk_st['RandomForest']} on student-test Top-1, with a mild over-fit. The largest gap is XGBoost's ({tv.loc['XGBoost', 'train_minus_cv_top1']:+.3f})."],
    ["Stability across validation folds", "Top-1 in each of the three trip-grouped folds",
     f"The spread over folds is similar for every model (standard deviation about {sd:.3f}) because the folds contain different trips. Random Forest is first in {int((fold_rank.loc['RandomForest'] == 1).sum())} of the 3 folds and never more than {behind:.3f} behind the best model of a fold."],
    ["Probability quality", "Log loss on student-test (Figure 10)",
     f"Random Forest {cmp_.loc['RandomForest', 'st_logloss']:.4f}, rank {rank_ll} of {n_models} (lowest: {ll_sorted.index[0]} {ll_sorted.iloc[0]:.4f}; highest: {ll_sorted.index[-1]} {ll_sorted.iloc[-1]:.3f})."],
    ["Inference speed", "Seconds to score 100,000 candidate rows (Table 7)",
     f"Random Forest {rf.infer_s_per_100k_rows:.3f} s. Fastest: {inf_sorted.index[0]} ({inf_sorted.iloc[0]:.3f} s); slowest: {inf_sorted.index[-1]} ({inf_sorted.iloc[-1]:.3f} s).{pipe_txt}"],
    ["Model complexity", "Size and settings of the model",
     f"The initial forest has 50 trees of depth up to 16; the tuned one has {bp['n_estimators']} trees of depth up to {bp['max_depth']}. The final bundle (tuned forest plus decoder settings) is {BUNDLE_MB:.0f} MB. Larger than a linear model or one tree, but small enough to ship in the repository."],
    ["Compatibility with the sequential decoder", "Whether the model's outputs suit the HMM",
     "The HMM uses the logarithm of each candidate's probability (normalised over the candidates of one GPS point) as its evidence, so probability quality matters more than the hard 0/1 decision. A forest outputs the average of its trees' leaf frequencies, which is consistent with its low log loss. "
     "We ran the HMM only with the selected model, so we cannot say how the others would behave in it."]]
sel_t = tbl(pd.DataFrame(sel_rows, columns=["Criterion (models guideline, section 6)", "What we measured", "Result for Random Forest"]), "Table 7c. Model-selection criteria and how Random Forest scored on each.")
add(f"""<h2>3. Model Experimentation and Comparison</h2>
<h3>3.0 Experimental design</h3>
<p>The task for the classifier is: given a GPS point and one of its (about {f_tr['rows_per_obs']:.0f}) candidate arcs, estimate the probability that this arc is the right one. We kept the design simple and fixed it before looking at any result.</p>
<ul>
<li><b>Fit data:</b> the {DSUM['train']['trips']} usable train trips. <b>Cross-validation:</b> 3 folds <i>grouped by trip</i>, so a trip never appears in both the training and the validation part (neighbouring GPS points are almost copies of each other, and splitting a trip would leak).
To save time, cross-validation fits use a fixed random 30% of the GPS points of the training folds (with all their candidates) and are scored on the full held-out fold. The final refit of each model uses all usable train rows.</li>
<li><b>Development comparison:</b> the 10 student-test trips, scored once per model after the final refit. Seeds are fixed (seed 0) and every configuration is saved in <code>reports/model_configs.json</code>.</li>
<li><b>Primary selection metric: observation-level Top-1 accuracy</b>: for each GPS point take the candidate with the highest score and check whether it is the reference arc. We chose it because it measures what we actually want (the right road for each point), needs no probability threshold,
and is not distorted by the large number of wrong candidates. A weakness: it ignores how confident the probabilities are, so we also report log loss, which matters because the HMM uses the probabilities directly.</li>
<li><b>Other metrics:</b> macro-F1 over the candidate classes at threshold 0.5, precision and recall of the correct-arc class, log loss, route continuity (the share of transitions between consecutive different predicted arcs that are connected within three directed road segments), confusion matrices, and training and inference time. The <i>candidate miss rate</i> (how often the reference road is not among the candidates) belongs to candidate generation, not to a model; see the note under Table 7b.</li>
</ul>
<h3>3.1 Overall comparison</h3>
{t31}{t31b}
<p>How to read it: the Top-1 scores of the top seven models differ by less than one cross-validation standard deviation (the standard deviation over folds is about {sd:.3f}, because the three folds contain different trips and fold 3 is easier than fold 1). So we do not claim large differences.
Random Forest has the best cross-validation Top-1 on average ({f4(cmp_.loc[best_cv,'cv_top1_mean'])}), although it is not first in every single fold (XGBoost is slightly ahead in two of the three folds and the neural network in one), and it has the best student-test Top-1, F1 and log loss; that is why we selected it. Gaussian NB is clearly the weakest.</p>
<h3>3.2 The nine models one by one</h3>
<p class='small'>Question numbers follow Deliverables.pdf (it numbers the questions 1, 2, 5, 6, 7, 8, 9). Each answer uses our own experiment results.</p>
{per_model}
<h3>3.3 Model visualisations and model selection</h3>
<p class='small'>Every one of the nine models appears in Figures 8 to 12. Figure 13 (neural network, per epoch) and Figure 14 (LightGBM and XGBoost, per boosting round) are the only training histories our models produce; no curve was made up for the others.</p>
{pair([(R/'figures'/'model_cv_macro_f1.png', 'Figure 8. Cross-validation macro-F1 per model, with the spread over folds (partition train).'), (R/'figures'/'model_student_top1_f1.png', 'Figure 9. Student-test Top-1 and macro-F1 per model (partition student_test).')])}
{pair([(R/'figures'/'model_logloss.png', 'Figure 10. Log loss per model on student-test (log scale; lower is better).'), (R/'figures'/'model_overfitting.png', 'Figure 11. Training vs cross-validation vs student-test Top-1: the over-fitting evidence.')])}
{fig(R/'figures'/'model_confusion_matrices.png', 'Figure 12. Student-test confusion matrices at threshold 0.5, row-normalised with counts (partition student_test). "Positive" = the candidate is the reference arc.', '86%')}
{pair([(R/'figures'/'model_mlp_loss_curve.png', 'Figure 13. Training and validation loss of the neural network per epoch. Only the MLP among our nine models is trained epoch by epoch, so this is the only per-epoch curve of that kind.'), (R/'figures'/'model_boosting_curves.png', 'Figure 14. Per-round log loss of LightGBM and XGBoost on a separate trip-level 80/20 split of the train trips (not the cross-validation folds).')])}
<p><b>What the pictures say.</b> The boosted models over-fit most: their validation loss is lowest at round 58 (LightGBM) and round 53 (XGBoost) of 300 and then drifts up (Figure 14), and XGBoost's training Top-1 ({f3(tv.loc['XGBoost','train_top1'])}) is far above its cross-validation Top-1 ({f3(tv.loc['XGBoost','cv_top1_mean'])}).
Early stopping would probably help them; we did not tune them because the forest was already ahead on the pre-agreed metric. The linear and probabilistic models barely over-fit but are also the least accurate. The forest sits in the middle: a mild gap and the best probabilities.</p>
<h3>Why Random Forest</h3>
{sel_t}
<p><b>Decision.</b> We selected the Random Forest because it had the best cross-validation and student-test Top-1, the best student-test Macro F1 and log loss, and a mild over-fit, at a prediction cost that is easily affordable.
The margin over XGBoost and the neural network is small, so the choice rests on the combination of criteria and not on one large win, and the ranking could change with other trips.</p>""")

# ------------------------------------------------------------------ 4 Bayesian
tt = trials.sort_values("number")
trial_t = tbl(pd.DataFrame({"Trial": tt.number, "Phase": ["random start-up" if k < 5 else "guided by the sampler" for k in tt.number], "Trees": tt.params_n_estimators, "Max depth": tt.params_max_depth,
                            "Min leaf": tt.params_min_samples_leaf, "Max features": tt.params_max_features.map(lambda v: f"{v:.2f}"), "Bootstrap share": tt.params_max_samples.map(lambda v: f"{v:.2f}"),
                            "CV Top-1": tt.value.map(f4), "Seconds": tt.user_attrs_seconds.map(lambda v: f"{v:.0f}")}), "Table 8. All nine Optuna trials (objective: mean cross-validation Top-1, higher is better).")
tvu = tuned
cvs = tcv.groupby("variant").agg(f1=("macro_f1", "mean"), f1sd=("macro_f1", "std"), top1=("top1", "mean"), ll=("logloss", "mean"))
paired = tcv.pivot_table(index="fold", columns="variant", values="macro_f1")
fold_gain = paired["tuned"] - paired["untuned"]
cmp_rows = []
for lab, key in (("Untuned Random Forest", "untuned"), ("Tuned Random Forest", "tuned")):
    r, c = tvu.loc[key], cvs.loc[key]
    cmp_rows.append([lab, f"{c.f1:.4f} ± {c.f1sd:.4f}", f4(c.top1), f4(r.st_top1), f4(r.st_macro_f1), f4(r.st_logloss), f"{r.infer_s_per_100k_rows:.3f} s", f"{r.fit_s_full:.0f} s"])
tune_cmp = tbl(pd.DataFrame(cmp_rows, columns=["Model", "CV Macro F1-score (mean ± std)", "CV Top-1", "Student-test Top-1", "Student-test Macro F1-score", "Student-test log loss", "Inference time (per 100k rows)", "Fit time (all rows)"]),
               "Table 9. Untuned vs tuned Random Forest: same folds, same fit subsample, same partitions, same metrics. CV Macro F1 is from scripts/04b_tuned_cv_macro_f1.py (it reproduces the untuned values of Table 7 exactly).")
d_top1, d_f1, d_ll = tvu.loc["tuned", "st_top1"] - tvu.loc["untuned", "st_top1"], tvu.loc["tuned", "st_macro_f1"] - tvu.loc["untuned", "st_macro_f1"], tvu.loc["tuned", "st_logloss"] - tvu.loc["untuned", "st_logloss"]
add(f"""<h2>4. Bayesian Hyper-parameter Optimisation (Random Forest)</h2>
<p>After the comparison we tuned the model that came first, the Random Forest, with Optuna's <b>TPE sampler</b>, a Bayesian method: it keeps a model of which parameter regions gave good scores and proposes new trials there, instead of trying settings blindly.
The questions below are those of the Deliverables document.</p>
<ol>
<li><b>Q1. Which hyper-parameters were searched?</b> The number of trees, the maximum tree depth, the minimum number of samples per leaf, the share of the 18 features tried at each split (max_features), and the share of rows drawn for each tree (max_samples).</li>
<li><b>Q2. Ranges and distributions.</b> Trees: integer 30 to 80. Depth: integer 10 to 20. Minimum leaf: integer 10 to 200 on a log scale (larger leaves smooth the probability estimates). Features per split: uniform 0.2 to 1.0 (the untuned forest used the square root, about 0.24). Bootstrap share: uniform 0.3 to 0.7.
The untuned values were 50 trees, depth 16, minimum leaf 50, square-root features and all rows per tree. We kept the ranges modest because every trial means three forest fits.</li>
<li><b>Q3. Objective.</b> Maximise the mean observation-level Top-1 over the same three trip-grouped folds and the same fixed 30% fit subsample as the comparison, so tuned and untuned cross-validation scores are directly comparable. This is the metric we had fixed for model selection. Student-test was not used to choose parameters.</li>
<li><b>Q4. Number of trials.</b> {ob['n_trials']} trials (5 random start-up trials, then 4 guided by the sampler), each with three forest fits. Computational cost: {ob['tuning_seconds'] / 60:.0f} minutes in total. We stopped there because of the time budget. With so few guided trials the search could do little more than random search would, and we say so.</li>
<li><b>Q5. Best values.</b> {bp['n_estimators']} trees, maximum depth {bp['max_depth']}, minimum leaf {bp['min_samples_leaf']}, {bp['max_features']:.2f} of the features per split, bootstrap share {bp['max_samples']:.2f}; cross-validation Top-1 {f4(ob['best_cv_top1'])}.</li>
<li><b>Q6. Did the tuned model outperform the untuned one?</b> Answer after the tables and figures below.</li>
</ol>
{trial_t}{tune_cmp}
{pair([(R/'figures'/'model_optuna_trials.png', 'Figure 15. Cross-validation Top-1 of each trial, and the best so far (partition train).'), (R/'figures'/'model_optuna_params.png', 'Figure 16. Each tuned parameter against the score it produced. With nine trials this is a scatter, not a parameter-importance analysis.')])}
<p><b>Q6, answer.</b> In cross-validation, slightly and consistently: Macro F1 {cvs.loc['tuned', 'f1']:.4f} against {cvs.loc['untuned', 'f1']:.4f} and Top-1 {cvs.loc['tuned', 'top1']:.4f} against {cvs.loc['untuned', 'top1']:.4f}; the tuned forest is ahead in each of the three folds (Macro F1 gains of {fold_gain.min() * 100:.2f} to {fold_gain.max() * 100:.2f} points).
On the student-test trips there is no gain: Top-1 {tvu.loc['tuned', 'st_top1']:.4f} against {tvu.loc['untuned', 'st_top1']:.4f} ({d_top1 * 100:+.2f} points), Macro F1 {tvu.loc['tuned', 'st_macro_f1']:.4f} against {tvu.loc['untuned', 'st_macro_f1']:.4f} ({d_f1 * 100:+.2f} points), log loss {tvu.loc['tuned', 'st_logloss']:.4f} against {tvu.loc['untuned', 'st_logloss']:.4f}.
Inference is slower ({tvu.loc['tuned', 'infer_s_per_100k_rows']:.2f} s against {tvu.loc['untuned', 'infer_s_per_100k_rows']:.2f} s per 100,000 rows) and fitting on all rows is faster ({tvu.loc['tuned', 'fit_s_full']:.0f} s against {tvu.loc['untuned', 'fit_s_full']:.0f} s).
Every trial landed between {tt.value.min():.4f} and {tt.value.max():.4f}, so the forest is not sensitive to these settings. We therefore claim a small gain on the data the search used and no gain on held-out trips. We kept the tuned forest in the final system because our pre-agreed rule (cross-validation Top-1) favours it and student-test shows a tie, not because it was clearly better.</p>""")

# ------------------------------------------------------------------ 5 HMM
diag_t = diag.sort_values("trip")
diag_t = tbl(pd.DataFrame({"Trip": diag_t.trip, "Labelled points": diag_t.n_eligible, "Both right": diag_t.both_right, "Only classifier right": diag_t.clf_right_hmm_wrong,
                           "Only HMM right": diag_t.clf_wrong_hmm_right, "Both wrong": diag_t.both_wrong}), "Table 12. Where the HMM helps and where it hurts, in points (partition student_test, tuned decoder).")
res_rows = []
for lab, r in (("Classifier only", None), ("Classifier + HMM, default decoder parameters", DA), ("Classifier + HMM, tuned decoder parameters", HA)):
    if r is None:
        res_rows.append([lab, pc(HA.pointwise_acc_classifier, 2), pc(HA.route_continuity_classifier, 2), pc(HA.route_f1_classifier, 2)])
    else:
        res_rows.append([lab, pc(r.pointwise_acc_hmm, 2), pc(r.route_continuity_hmm, 2), pc(r.route_f1_hmm, 2)])
res_t = tbl(pd.DataFrame(res_rows, columns=["Method", "Pointwise accuracy", "Route continuity", "Route F1"]), "Table 10. Classifier only vs classifier + HMM, observation-weighted over the ten student-test trips (final tuned Random Forest).")
pt = ct.copy()
if lat is not None:
    pt = pt.merge(lat[["trip", "pipeline_ms"]], on="trip", how="left")
pt_t = tbl(pd.DataFrame({"Trip": pt.trip, "GPS points": pt.n_obs, "Accuracy: classifier": pt.pointwise_acc_classifier.map(lambda v: pc(v)), "Accuracy: + HMM": pt.pointwise_acc_hmm.map(lambda v: pc(v)),
                         "Continuity: classifier": pt.route_continuity_classifier.map(lambda v: pc(v)), "Continuity: + HMM": pt.route_continuity_hmm.map(lambda v: pc(v)),
                         "Route F1: classifier": pt.route_f1_classifier.map(lambda v: pc(v)), "Route F1: + HMM": pt.route_f1_hmm.map(lambda v: pc(v))}), "Table 11. Per-trip results (partition student_test, tuned decoder).")
t7txt = ""
if t7:
    t7txt = (f"On trip 00000007 the HMM is wrong where the classifier is right at {t7['n_points']} points, in {t7['n_ranges']} short stretches of at most {t7['max_len']} points. All of them fall in the last part of the trip "
             f"(from observation {t7['first_obs']} of {t7['n_obs']} onward), where the vehicle is moving more slowly (median {t7['speed_bad']:.1f} m/s on those points against {t7['speed_other']:.1f} m/s on the rest of the trip, and none below 1 m/s, so it is not stopped). "
             f"In every one of these stretches the road the HMM chose is connected to the reference road: it picked a neighbouring road, not a disconnected jump. Our reading: when the vehicle drives slowly in a built-up area, "
             "adjacent roads are all about equally compatible with the small distance travelled, and the small transition scale then trusts the GPS geometry more than the classifier. We did not test this explanation further.")
add(f"""<h2>5. HMM and Viterbi route reconstruction</h2>
<h3>5.1 How route reconstruction works (Q1 to Q5)</h3>
<p><b>Q1. Hidden states.</b> For every GPS point, one state per candidate arc. A state is therefore a real road segment, and the link between a candidate and its road is never lost.</p>
<p><b>Q2. Emission probabilities (how well a state explains its GPS point).</b> The classifier's probability for that candidate, divided by the sum over the point's candidates so they add up to one, and raised to a power α: <code>log e = α · log(p / Σp)</code>.</p>
<p><b>Q3. Transition probabilities (how plausible it is to go from candidate i at one point to candidate j at the next).</b> Following Newson and Krumm (2009), we compare two distances: the straight-line distance <i>d<sub>gps</sub></i> between the two GPS points,
and the distance <i>d<sub>route</sub></i> along the road network between the two candidate positions (forward only, along directed arcs, found with a shortest-path search on a small graph around the trace; on the same arc it is the difference in position, with 15 m of tolerance for GPS jitter).
A vehicle that moved 20 m should sit on candidates about 20 m apart <i>along the road</i>. <code>log T = −| d<sub>gps</sub> − d<sub>route</sub> | / β</code>.</p>
<p><b>Viterbi decoding.</b> Start with <code>δ₁(j) = emit₁(j)</code>. For each next point: <code>δₖ₊₁(j) = emitₖ₊₁(j) + maxᵢ [ δₖ(i) + log Tₖ(i, j) ]</code>, remembering the best i. At the end take the best final state and walk back. The work is dominated by the network distances, which depend only on the candidates, so they are computed once per trip.</p>
<p><b>Q4. Missing candidates and large GPS gaps.</b> A GPS point with no candidate at all is dropped from the chain and reported as unmatched. If two consecutive points are more than 15 s or 500 m apart, the transition becomes uniform: the chain restarts from the emissions alone, instead of forcing an unreliable connection across a gap.</p>
<p><b>Q5. Consecutive points that suggest disconnected road segments.</b> If no directed path exists within the search limit (3 × <i>d<sub>gps</sub></i> + 300 m), that pair gets a fixed penalty (never minus infinity, so decoding can always continue). Choosing a disconnected pair is therefore possible, but expensive: Viterbi takes it only if the emissions strongly demand it. After decoding, any remaining break is bridged with the shortest path when the route is drawn.</p>
<h3>5.2 Tuning the decoder</h3>
<p>The three decoder parameters (emission power α, transition scale β, unreachable-pair penalty) were tuned with Optuna (60 trials) on 30 random train trips, using <b>out-of-fold</b> classifier probabilities, so that the emissions were as noisy as they would be on unseen trips.
On that sample, observation-level Top-1 was {pc(hb['classifier_only_sample_top1'], 2)} for the classifier alone, {pc(hb['default_sample_top1'], 2)} with default parameters and {pc(hb['best_sample_top1'], 2)} with the tuned ones.
The tuned values are α = {hb['alpha']:.2f}, β = {hb['beta']:.2f} m and penalty = {hb['invalid_pen']:.1f}. A small β means the decoder trusts agreement between GPS distance and road distance strongly. The gap limits (15 s, 500 m) were fixed, not tuned.</p>
<h3>5.3 Did HMM decoding improve the final route? (Q6)</h3>
<p class='small'><b>Definitions.</b> <i>Pointwise accuracy</i>: the share of labelled GPS points whose predicted road equals the reference road. <i>Route continuity</i>: of all transitions between consecutive different predicted roads, the share where the second road can be reached from the first within three directed road segments.
<i>Route F1</i> (sequence-level): F1 between the set of distinct roads in the predicted route and the set of roads in the reference route. <i>Candidate miss rate</i>: the share of labelled points whose reference road is not among the candidates ({pc(HA.candidate_miss_rate, 2)} on student-test, so candidate generation alone can cause at most that share of errors).</p>
{res_t}{fig(FIG_HMM, 'Figure 17. Pointwise accuracy per student-test trip, classifier only vs classifier + HMM.', '86%')}{pt_t}
<p><b>Q6, answer: yes, overall, and mostly through connectivity.</b> Accuracy rises by {100 * (HA.pointwise_acc_hmm - HA.pointwise_acc_classifier):.1f} points and route continuity from {pc(HA.route_continuity_classifier)} to {pc(HA.route_continuity_hmm)}: the classifier alone produces many disconnected jumps, the decoded route almost none.
Tuning the decoder added a further {100 * (HA.pointwise_acc_hmm - DA.pointwise_acc_hmm):.1f} points over the default parameters. Nine of the ten trips improve.</p>
<h3>5.4 Where it did not help</h3>
{diag_t}
<p>{t7txt}</p>
<p>Trip 00000077 is the hardest trip for both methods: {pc(float(ct[ct.trip=='00000077'].pointwise_acc_classifier.iloc[0]))} for the classifier and {pc(float(ct[ct.trip=='00000077'].pointwise_acc_hmm.iloc[0]))} with the HMM, and in {int(diag[diag.trip=='00000077'].both_wrong.iloc[0])} points both are wrong, so the classifier's evidence itself is weak there.
The HMM can reorder and reconnect evidence but cannot create evidence that is not there.</p>""")

# ------------------------------------------------------------------ 6 Folium
req6 = pd.DataFrame([
    ["Original GPS observations", "Layer 1, blue dots (<code>1 Original GPS observations</code>)."],
    ["Relevant road-network geometry", "Layer 2, grey lines: the road segments that were candidates for at least one GPS point (not the whole network, which has up to 300,000 arcs)."],
    ["Reference route", "Layer 3, wide translucent green line. Offline report only; the evaluator interface never shows it."],
    ["Predicted route", "Layer 4, classifier-only route (orange dashed) with red circles at its disconnected jumps; layer 5, HMM / Viterbi route (purple) with a green start marker and a red end marker."],
    ["Distinguishable colours and markers", "Each layer has its own colour and line style; markers are used only for jumps and for the start and end of the HMM route."],
    ["Labels or a legend", "A legend box (bottom left) names every colour; a title box (top left) shows the trip identifier and its real scores from the run."],
    ["Extent and zoom", "The view is fitted to the trace (<code>fit_bounds</code>). A second file per trip (<code>_zoom</code>) is fitted to the 80-observation window where the two methods disagree most."],
    ["Easy comparison of observed GPS and predicted route", "All layers share one map and can be switched on and off; the GPS dots lie along the purple line where the HMM is right."],
    ["Layer controls", "A Folium layer control, open by default, with the five layers above."]], columns=["Requirement (Deliverables, section 6)", "How the maps meet it"])
add(f"""<h2>6. Geographical Visualisation Using Folium</h2>
<p>All maps are generated from the team's own implementation: the final model bundle is run on student-test trips through the same pipeline code the API uses, and the maps are drawn from its output (<code>scripts/10_folium_prediction_maps.py</code>; the map of a raw trip in Section 2 comes from <code>scripts/08_eda_figures.py</code>).
Nothing was drawn by hand. The interactive files are committed in <code>reports/maps/</code>: <code>pred_trip_&lt;id&gt;.html</code> and <code>pred_trip_&lt;id&gt;_zoom.html</code> for trips 00000052, 00000009, 00000040, 00000007 and 00000077, and <code>eda_trip_00000085.html</code>.
We chose trips where the HMM clearly helps (00000009, 00000040, 00000052), the trip where it hurts (00000007) and the hardest trip (00000077). The figures below are screenshots of those interactive maps.</p>
{tbl(req6, "Table 14. How the maps meet the Folium requirements.")}
{pair([(R/'screenshots'/'map_pred_00000009.png', 'Figure 18. Trip 00000009: the HMM follows the reference route where the classifier-only route (orange) wanders (partition student_test).'), (R/'screenshots'/'map_pred_00000007.png', 'Figure 19. Trip 00000007, the trip where the HMM scores lower than the classifier. At this scale the routes are almost identical (partition student_test).')])}
{pair([(R/'screenshots'/'map_pred_00000007_zoom.png', 'Figure 20. Zoom into the window of trip 00000007 with most disagreement: the HMM picks a neighbouring road.'), (R/'screenshots'/'map_pred_00000077_zoom.png', 'Figure 21. Zoom into trip 00000077, the hardest trip, where both methods are wrong at many points.')])}""")

# ------------------------------------------------------------------ 7 API / UI / deployment
LOG = (R / "api_console_log.txt").read_text(encoding="utf-8").strip()    # real lines copied from the API console (see scripts note in the caption)
lat_t = ""
if lat is not None:
    lt = lat.sort_values("trip")
    lat_t = tbl(pd.DataFrame({"Trip": lt.trip, "GPS points": lt.gps_points, "Candidates + features (ms)": lt.candidates_ms.map(lambda v: f"{v:.0f}"), "Classifier (ms)": lt.classifier_ms.map(lambda v: f"{v:.0f}"),
                              "HMM (ms)": lt.hmm_ms.map(lambda v: f"{v:.0f}"), "Pipeline total (ms)": lt.pipeline_ms.map(lambda v: f"{v:.0f}")}),
                "Table 13. Server-side time per trip, measured by running the final bundle on the ten student-test trips (laptop otherwise idle). Scoring against the reference route and building the response add a little on top.")
add(f"""<h2>7. API, user interface and deployment</h2>
<h3>7.1 The API (server laptop)</h3>
<p>The server is a FastAPI application with three endpoints. <code>GET /health</code> returns the model name and the decoder settings. <code>GET /trips</code> lists the trips the server can serve (identifier, number of GPS points, duration).
<code>POST /predict</code> with <code>{{"trip_id": "..."}}</code> runs the whole pipeline on that trip and returns the GPS points, the road network near the trace, the classifier-only route with its disconnected jumps, the HMM route and the aggregate scores.</p>
<ul>
<li><b>Ground truth stays on the server.</b> The reference route is read only after the prediction exists, only to compute accuracy, continuity and F1, and it is never part of any response. We checked this for all ten trips: no response field is derived from the reference route other than the aggregate numbers.</li>
<li><b>No hardcoded results.</b> Every request loads the trip files and runs candidates, classifier and HMM from scratch; nothing is cached or looked up. The client sends only the trip identifier.</li>
<li><b>Safe by construction.</b> A request can only name a trip folder that the server itself found, which rules out path tricks. Browsers are accepted only from localhost and private network addresses (CORS). A broken trip folder is skipped without stopping the server; unreadable input answers 422 with a plain message; unexpected failures answer 500 without exposing file paths; and if scoring is impossible, the prediction is still returned.</li>
<li><b>One request at a time.</b> Predictions are serialised by a lock, which is simple and safe for one evaluator. End to end, the server needed between about 0.3 s and 6 s per trip in our test (the console lines in Figure 22 show the server time of each request); the largest trip, 00000056 with almost 4,000 GPS points, was slowest (Table 13).</li>
<li><b>Console evidence.</b> The server prints one line per request with the trip, the client's address, the number of points, the accuracy and continuity of both methods, and the server time (Figure 22).</li>
</ul>
{lat_t}
<pre>{html.escape(LOG)}</pre>
<p class='small'>Figure 22. Real console output of the API (one <code>PREDICT</code> line per request) while all ten student-test trips were requested one after another from the same laptop (client 127.0.0.1). The lines are copied unchanged from the server console into reports/api_console_log.txt. In the two-laptop test the same line shows the second laptop's address instead.</p>
<p><span class='todo'>Insert here: screenshot of the server laptop's console while the second laptop requests trips (the brief asks for a screenshot of the receiving laptop's console).</span></p>
<h3>7.2 The user interface (client laptop)</h3>
<p>The interface is a web page built with React and TypeScript. It opens with a boot screen that checks the server (<code>/health</code> and <code>/trips</code>), then an optional scroll-driven introduction that explains the pipeline using one real prediction, and then the main screen:
the ten trips on the left, the map in the middle and the results on the right. After a trip is run, the route is drawn on top of an OpenStreetMap street map (the map tiles are fetched by the browser; with no internet the interface says so and falls back to an offline road-network drawing),
and the right panel shows pointwise accuracy, route continuity and route F1 for the classifier alone and with the HMM, the candidate miss rate, the number of disconnected jumps and the latency split. A request log shows every call the interface makes.
The interface contains no model, no data and no stored results: it only displays what the API returns. It handles errors (server unreachable, unknown trip, slow server) with readable messages and a retry box.</p>
{pair([(R/'screenshots'/'ui_01_connect.png', 'Figure 23. Boot screen: it checks the server and loads the trip list.'), (R/'screenshots'/'ui_02_intro_0.5.png', 'Figure 24. Scroll introduction (stage 3): the classifier-only route with its disconnected jumps (red rings).')])}
{pair([(R/'screenshots'/'ui_03_trip_list.png', 'Figure 25. Main screen: the ten trips and the Reconstruct button.'), (R/'screenshots'/'ui_05_error_state.png', 'Figure 26. Error state: the server cannot be reached; the interface says which address it tried and offers a retry.')])}
{fig(R/'screenshots'/'ui_04_result_basemap.png', 'Figure 27. A finished run: the HMM route (gold) on the street map, with the classifier-only and HMM scores, the latency split and the request log on the right.', '92%')}
<h3>7.3 Deployment</h3>
<p>We ran the real two-laptop set-up: the server laptop runs the API, and the interface runs on a second laptop connected to the server laptop's Wi-Fi hotspot. The second laptop called the API across the network and displayed the returned routes and scores. The client laptop needs only Python 3 and a small package (the built interface plus a start script);
it needs internet only for the street-map tiles. On the server laptop the API's port has to be allowed through the firewall, and the folder with the evaluation trips is set with one environment variable. Step-by-step instructions, with a 60-second connection check and a table of likely problems, are in <code>RUNBOOK.md</code> in the repository.</p>
<p>Before the demonstration we also had the API and the interface checked for crashes and wrong behaviour with unusual input (trips with one GPS point, missing or inconsistent files, odd identifiers, slow or silent servers). Most findings were about robustness, for example a single malformed trip folder preventing the server from starting; they are fixed and re-tested.</p>
<p><b>Code that matches the demonstration.</b> The results in this report were produced with the final model bundle committed at <code>artifacts/final/bundle.joblib</code>, which the API loads when it starts, so the demonstration runs the code and model of the repository at {REPO_URL}{(' (commit <code>' + GIT + '</code>)') if GIT else ''}.</p>""")

# ------------------------------------------------------------------ 8 integrity, appendices
repro = pd.DataFrame([
    ["Dataset inventory, data quality, candidate-radius coverage (Tables 1-5, Figures 3-7)", "scripts/01_eda_inventory.py, 08_eda_figures.py", "reports/eda_trip_summary.csv, reports/tables/eda_*, reports/figures/eda_*"],
    ["Labelled candidate tables", "scripts/02_build_dataset.py", "artifacts/data (local, not committed), reports/dataset_summary.json (made by scripts/13_build_report.py from those tables)"],
    ["Nine-model comparison (Tables 7, 7b, 7c, Figures 8-14)", "scripts/03_train_models.py, 09_model_figures.py, 11_train_accuracy.py, 12_boosting_curves.py", "reports/model_comparison.csv, cv_fold_results.csv, train_vs_cv_vs_student.csv, boosting_curves.json, model_configs.json"],
    ["Bayesian optimisation (Tables 8-9, Figures 15-16)", "scripts/04_tune_random_forest.py, 04b_tuned_cv_macro_f1.py", "reports/optuna_trials.csv, optuna_best.json, optuna_rf.db, tuned_vs_untuned.csv, tuned_vs_untuned_cv.csv"],
    ["Decoder tuning and HMM comparison (Tables 10-12, Figure 17)", "scripts/07_tune_decoder.py, 10b_hmm_diagnostics.py", "reports/hmm_best.json, hmm_trials.csv, classifier_vs_hmm*.csv, hmm_diagnostics*.csv"],
    ["Folium maps (Figures 6, 18-21)", "scripts/08_eda_figures.py, 10_folium_prediction_maps.py", "reports/maps/*.html, reports/screenshots/map_*.png, eda_map.png"],
    ["Final inference bundle, API, interface (Section 7, Table 13, Figure 22)", "scripts/06_export_bundle.py, pathsnap/api.py, web/", "artifacts/final/bundle.joblib, RUNBOOK.md, reports/latency_final_bundle.csv, reports/api_console_log.txt"],
    ["This report", "scripts/13_build_report.py (text in 13b_report_text.py)", "report/PathSnap_Technical_Report.html and .pdf"]], columns=["What", "Script(s)", "Output file(s)"])
rules8 = pd.DataFrame([
    ["The report must represent the team's actual work and understanding.",
     "Every experiment in this report was run in the repository by the scripts listed in Appendix B, on the train and student-test partitions only. Each number in the tables and text is read from the saved result files by <code>scripts/13_build_report.py</code>. "
     "<span class='todo'>Team to confirm before submission: every member has read the report and can explain it.</span>"],
    ["No copying from other teams.",
     "All code, models and results were produced inside this project. <span class='todo'>Team to confirm: no model files, code, configurations or results were shared with or taken from another team.</span>"],
    ["AI tools must not replace experimentation or analysis.",
     "The training, tuning, decoding, API and interface tests were actually executed and their outputs are in the repository. The AI tool wrote and ran code under our direction. The comparisons and explanations in this report are drawn from those outputs; where an explanation is our interpretation (for example, why the HMM loses on trip 00000007), the text says it was not tested further."],
    ["Disclose AI assistance.",
     "We used <b>Claude Code (Anthropic)</b> for: writing and running the data-processing, feature, model, tuning, HMM, API and Folium code; building the web interface; generating the figures; reviewing the code for bugs (several automated review passes found problems that we fixed, for example a malformed trip folder preventing the server from starting); "
     "and writing the first draft of this report. The team made the key design decisions: which model families to compare, observation-level Top-1 as the selection metric, a Newson and Krumm style HMM, labels from a monotone alignment to the reference route, and the use of the evaluator-facing design. "
     "<span class='todo'>Team to edit this paragraph so that it matches the event's AI-disclosure policy and describes your own contribution accurately. The interface's visual style was inspired by a website the team admired (edolus.com); confirm how you want that credited.</span>"],
    ["Every team member must be able to explain the code, experiments, results and conclusions.",
     "<span class='todo'>Team to confirm and, if useful, add who explains which part.</span> The repository's <code>team_packages/</code> folder contains one short brief per part of the system, written for this purpose."]],
    columns=["Rule (Deliverables, section 8)", "Our statement"])
add(f"""<h2>8. Academic Integrity, Plagiarism and AI-Use Disclosure</h2>
<p>The table answers each rule of section 8 of the Deliverables document. Results can be verified by running the scripts in Appendix B against the committed files: <a href='{REPO_URL}'>{REPO_URL}</a>{(' (commit <code>' + GIT + '</code>)') if GIT else ''}.
No data from the evaluator partition was read at any time, ground-truth routes never leave the server, and no prediction is hard-coded (the API runs the full pipeline on every request).</p>
{tbl(rules8, "Table 15. Academic-integrity statements.")}

<h2>Appendix A. Limitations and next steps</h2>
<ul>
<li><b>The reference is not perfect truth.</b> Labels come from aligning points to a supplied route. Two training trips had routes that do not match their own GPS traces and were excluded; similar problems could exist in the hidden trips and would lower any score without the model being at fault.</li>
<li><b>The comparison is small.</b> Ten student-test trips and three cross-validation folds cannot separate models whose scores differ by less than a point, and trips differ a lot in difficulty. The nine-model ranking should be read as "the top few are about equal".</li>
<li><b>Light tuning.</b> The Bayesian search ran only nine trials and the gains were small. The boosted models were not tuned at all, although their curves suggest early stopping would help.</li>
<li><b>By design, the classifier looks at one point at a time</b> (plus a short heading window). A model that also sees its neighbours would be stronger on its own, but would overlap with what the HMM does; we kept them separate so that the HMM's contribution can be measured.</li>
<li><b>The HMM has a few weak spots.</b> It can prefer a neighbouring road at low speed (trip 00000007), and it cannot repair trips where the classifier evidence is poor (trip 00000077). Its transition model is distance-based; road type or speed limits, which the data does not contain, could improve it. It was evaluated only with the selected classifier.</li>
<li><b>Speed.</b> The server handles one request at a time and the largest trips take several seconds, which is fine for a demonstration but not for many users.</li>
<li><b>Hidden data.</b> The evaluator trips are unseen, so the numbers here are development results and the demonstration numbers will differ.</li>
</ul>
<p><b>Next steps:</b> early stopping and a larger search for the boosted models; learning the decoder parameters per trip type; adding road-class information if available; scoring several candidate routes and returning their confidence.</p>

<h2>Appendix B. Reproducibility and software</h2>
<p>Random seeds are fixed (seed 0) and the settings of every model are saved. Set <code>PATHSNAP_DATA_ROOT</code> to the folder holding the <code>train</code> and <code>student_test</code> partitions, then run the scripts in the order of the repository README. The table says which script produces which result.</p>
{tbl(repro)}
<p class='small'><b>Software.</b> Python 3.11 with {SOFTWARE_PY}. Interface: {SOFTWARE_JS}, built with Vite and TypeScript. Street-map tiles and data: © OpenStreetMap contributors.</p>

<h3>References</h3>
<ol class='small'>
<li>P. Newson and J. Krumm. Hidden Markov map matching through noise and sparseness. <i>Proc. 17th ACM SIGSPATIAL GIS</i>, 2009.</li>
<li>L. Breiman. Random forests. <i>Machine Learning</i> 45(1), 2001.</li>
<li>T. Akiba, S. Sano, T. Yanase, T. Ohta and M. Koyama. Optuna: a next-generation hyperparameter optimization framework. <i>Proc. KDD</i>, 2019.</li>
<li>F. Pedregosa et al. Scikit-learn: machine learning in Python. <i>JMLR</i> 12, 2011.</li>
<li>T. Chen and C. Guestrin. XGBoost: a scalable tree boosting system. <i>Proc. KDD</i>, 2016. G. Ke et al. LightGBM: a highly efficient gradient boosting decision tree. <i>NeurIPS</i>, 2017.</li>
</ol>""")

# ------------------------------------------------------------------ write and print
doc = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>PathSnap: technical report</title><style>{CSS}</style></head><body>" + "\n".join(parts) + "</body></html>"
HTML = OUT / "PathSnap_Technical_Report.html"
PDF = OUT / "PathSnap_Technical_Report.pdf"
HTML.write_text(doc, encoding="utf-8")
print(f"wrote {HTML} ({HTML.stat().st_size / 1e6:.1f} MB)")
if Path(CHROME).exists():
    prof = OUT / ".chrome_profile"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", f"--user-data-dir={prof}", "--no-pdf-header-footer", f"--print-to-pdf={PDF}", HTML.as_uri()],
                   check=False, timeout=240, capture_output=True)
    print(f"wrote {PDF} ({PDF.stat().st_size / 1e6:.1f} MB)" if PDF.exists() else "PDF was not produced")
else:
    print("Chrome not found; open the HTML in a browser and print it to PDF.")
