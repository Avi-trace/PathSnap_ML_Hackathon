# Piece 4: Client UI (report section 7, UI part)

## Your part
Own the evaluator-facing web interface that runs on Laptop 2: connect to the server, list the ten evaluation trips, run a selected trip through the API, and show the predicted route and results.

## Start here
1. Read **`client_handoff/CLIENT_README.md`**: it is your complete specification (API contract, hard rules, design brief, acceptance checklist, hand-in).
2. `client_handoff/mock_server.py` + `samples/` let you build without the real server (it flags itself as a mock; the UI must show a red banner for it).
3. `web/` is a **working skeleton** (Vite + React + TypeScript). You can extend it or rebuild from the spec. It already has: boot screen, 600vh scroll intro driven by a live API call, trip picker, offline canvas map, results panel, request log, error handling,
   WebGL post-processing with automatic fallback, and error boundaries. If you keep it, read every file until you can explain it: you will be asked.

Run: `cd web && npm install && npm run dev -- --host`, then open `http://localhost:5173/` with `client_handoff/mock_server.py` running on port 8000 (development), or `?api=http://<server-ip>:8000` against the real server. See `web/README.md`.

## What is already verified (honest status)
Built and type-checked; run in a headless browser against the live API on all ten student trips (numbers matched the API response); a few trips re-checked independently. **Not verified:** smoothness of the WebGL effects on a real GPU (a CPU-only browser switched them off automatically),
the reduced-motion layout, and the error-boundary message. Please test these on your laptop.

## Explain it (plain language)
The UI never computes the route: it calls `GET /health` and `GET /trips`, then `POST /predict {"trip_id"}` and draws what comes back (GPS points, road network, the classifier-only route with its disconnected jumps, the HMM route, and the scores).
Coordinates are `[lon, lat]`; the map is drawn on a canvas with a local equirectangular projection, so it works with no internet or map tiles. Ground truth never reaches the UI: the server computes the accuracy and sends only aggregate numbers.
The scroll intro is not canned: it fetches one real prediction and animates its layers with scroll position. The skeleton contains **no bundled mock data** (already removed): every number comes from the API. For development use `client_handoff/mock_server.py`; the UI shows a red banner when `/health` says `"mock": true`, and that server must never be used in the final demo.

## Your deliverables
1. The finished `web/` folder (source, `package.json`, lockfile, a short README with install/run/`?api=` usage; no `node_modules`, no `dist`), with the mock removed.
2. Pass the acceptance checklist in `client_handoff/CLIENT_README.md` section 9 against the **real** server (ask the server teammate for their IP).
3. Screenshots for the report: connect screen, trip list, a finished run (map + results), an error state, the scroll intro.
4. Design decisions already made by the team: keep the scroll intro and the WebGL effects. If the effects lag on the demo laptop, switch them off with `?fx=0` (the pass also switches itself off when frames are slow) and tell the team lead.

## Likely viva questions
What does the UI send and receive? Why is the reference route never shown? How is the map drawn without tiles? What happens if the API is down? How does the intro get its data? Why must mock code be removed?

## Hand back
The finished `web/` folder (zipped without `node_modules`) and the screenshots.
