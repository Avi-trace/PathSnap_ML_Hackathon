# Piece 1: Data & Features (report section 2.2)

## Your part
Own everything between the raw files and the table the models train on: inspection, preprocessing, coordinate handling, labels, candidate generation and features.
You answer: *How did you preprocess the data? What patterns did you see? What features did you extract and why?*

## What is already built (read these files)
| File | What it does |
|---|---|
| `pathsnap/io.py` | readers for the four raw file types (`.track .nodes .arcs .route`) |
| `pathsnap/geo.py` | per-trip UTM projection (metres) and point-to-segment distance |
| `pathsnap/candidates.py` | all arcs within a radius of each GPS point |
| `pathsnap/labels.py` | per-point reference label from the ordered route (monotone dynamic programming) |
| `pathsnap/features.py` | the 18 per-candidate features |
| `scripts/01_eda_inventory.py` | inventory, data-quality checks, candidate-radius coverage -> `reports/eda_trip_summary.csv` |
| `scripts/02_build_dataset.py` | builds the labelled candidate tables for all 90 trips |

## Explain it (plain language, with the real numbers)
**The data.** 90 trips: 80 `train`, 10 `student_test` (a third, evaluator, partition is hidden). Each trip is a folder of four files. `.track` = GPS rows (lon, lat, time).
`.nodes` = road intersections (lon, lat). `.arcs` = directed road segments between node ids. `.route` = the arc ids the vehicle actually used, in order.
A GPS row does **not** correspond to a node or arc row: the link is only geometric ("which arc is this point near?"). Total 227,577 GPS points.

**Coordinates.** Degrees are not distances. The trips are all over the world (25 UTM zones), so each trip is projected into its own UTM zone (metres, scale error < 0.1 %).

**Data-quality findings** (from `reports/eda_trip_summary.csv`): no NaN or impossible coordinates; timestamps always increase; 96.7 % of all steps are exactly 1 s apart (the per-trip share averages 94.3 %), 47 trips have a gap over 5 s
(largest 153,064 s) and 1,538 steps imply more than 50 m/s (GPS jumps or gaps); networks contain 24 self-loops and 1,903 duplicate arcs, 92 % of arcs have a reverse arc, 89 of 90 networks are split into several components;
reference routes are valid and nearly always head-to-tail connected (99.94 %); 10 trips revisit arcs.

**The labelling problem.** The data gives the route, not "which arc each point is on". We align points to the route with a dynamic programme that never moves backwards along the route and minimises point-to-arc distance
(cost capped at 200 m, a tiny penalty of 0.2 per route step discourages jumping ahead). A plain "nearest route arc" would fail on routes that revisit or run parallel to themselves. Points whose aligned arc is more than 150 m away get no label.

**Two bad trips.** `00000033` and `00000054` (12 % of train points) have GPS traces whose median distance to their own reference route is 122 m and 125 m with a wide spread, and shifting the trace by its mean offset does not fix them.
The distance histograms are **bimodal** (`reports/figures/eda_06_gps_route_offset.png`): part of each trace lies within 0-20 m of the route and a large part about 120-145 m away, i.e. the route and trace agree in places and are displaced elsewhere.
Treat the route as unreliable there (the brief warns routes are "reference", not perfect truth): they are built but excluded from fitting.

**Candidates.** For every GPS point take all arcs within 75 m (exact point-to-segment distance; long arcs are split into 40 m pieces for the spatial index so none are missed). Chosen by measurement on clean trips:
miss rate at 25 / 50 / 75 / 100 / 150 m = 2.83 / 1.14 / 0.67 / 0.42 / 0.28 %. At 75 m: 15.8 candidates per point, 2.88 M training rows, final miss rate 0.45 % (train) and 0.09 % (student-test).
A baseline that picks the nearest arc is right only 55.7 % of the time, because each road has two arcs at the same distance. This is why direction features matter.

**The 18 features** (all per candidate; `FEATURES` in `features.py`):
| Feature | Meaning | Why |
|---|---|---|
| `dist`, `dist_gap`, `dist_rank`, `n_cand` | distance to the arc, gap to the nearest candidate, rank by distance, number of candidates | closeness and local road density |
| `along_m`, `remain_m`, `arc_len` | position along the arc and arc length | where on the segment the point projects |
| `lateral` | signed side offset (left/right of the arc direction) | separates the two opposite arcs of one road |
| `cos_dh`, `abs_dh`, `cos_gap` | heading difference between the GPS direction (over +/-2 points) and the arc direction | direction of travel; 0 / 90 deg when the vehicle is not moving |
| `speed_w`, `step_speed` | speed over the +/-2 window / last step | slow points have unreliable heading |
| `dt_prev`, `dt_next` | time to neighbouring points | GPS gaps |
| `outdeg_end`, `indeg_start`, `two_way` | junction degree of the arc ends, whether a reverse arc exists | network structure |

Parameters: radius 75 m, label cut-off 150 m, heading window +/-2 points, "moving" threshold 4 m, max window 10 s.

## Your deliverables (the report asks for all of these, and for the partition used in each)
`scripts/08_eda_figures.py` already produces everything below (see `reports/figures/INDEX_eda.md`; tables in `reports/tables/`; Folium map in `reports/maps/`; screenshot in `reports/screenshots/eda_map.png`).
**Your job: run it, read the script until you can explain every plot, check each figure for correctness, extend it where you see gaps (more trips, a second Folium map, other plots), and write the report text yourself.** The items:
1. Dataset inventory table: trips per partition, files, observations per trip (from `eda_trip_summary.csv`).
2. Missing / invalid value table (NaN, implausible coordinates, non-increasing timestamps) and a short data-quality log of decisions.
3. Histogram of sampling intervals (log scale) and of trip lengths / durations.
4. GPS trajectory plots in projected metres for a few trips; GPS coordinate extent per trip.
5. Road-network plot for one trip + a connectivity table (nodes, arcs, components, largest component share).
6. Plot of the GPS-to-route offset for trips 33 and 54 versus a normal trip (evidence for excluding them).
7. **At least one Folium map** (mandatory): raw GPS (blue dots), road network (grey), reference route (green) for a train trip, with legend, layer control, sensible zoom. Save `reports/maps/eda_trip_<id>.html` and a screenshot.
8. The feature table above, in your own words, and the candidate-radius coverage table.

## Likely viva questions
Why project to UTM? How do you get a label per GPS point and why monotone? How was 75 m chosen? What does `dist_rank` do and why is nearest-arc only 55.7 % right? Why were two trips excluded? What are the limits of the reference route?

## Run it
`python scripts/01_eda_inventory.py --data-root "<data root>"`, then `python scripts/02_build_dataset.py --data-root "<data root>"`.
