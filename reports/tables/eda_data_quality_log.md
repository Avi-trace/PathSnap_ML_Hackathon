| issue | trips_affected | decision |
|---|---|---|
| step dt != 1 s (irregular sampling) | 66 | kept; dt_prev/dt_next features let the model see GPS gaps instead of assuming fixed-rate sampling |
| gap > 5 s between consecutive points (max 153064 s) | 47 | kept; only affects speed/heading features locally around the gap, points are still labelled independently |
| step implied speed > 50 m/s | 42 | kept as-is (no row dropped); flagged for the classifier via speed_w/step_speed features, not hand-cleaned |
| NaN GPS lon/lat/t | 0 | none found; no action needed |
| implausible coordinates (|lon|>180 or |lat|>90) | 0 | none found; no action needed |
| non-increasing timestamps | 0 | none found; no action needed |
| road network split into >1 connected component | 89 | kept; candidate generation is purely geometric (radius search) and does not require network connectivity |
| self-loop / duplicate arcs in network | 60 | kept (24 self-loops, 1903 duplicate arcs total); harmless for radius-based candidate search |
| GPS trace far (p50 > 100 m) from its own reference route | 2 | trips 00000033 and 00000054 identified and excluded from label fitting (route treated as unreliable); not dropped from the inventory/EDA |
