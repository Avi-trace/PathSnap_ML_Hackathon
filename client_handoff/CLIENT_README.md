# PathSnap: client (Laptop 2) build spec

**Audience:** a Claude Code session building the evaluator-facing UI, plus the human running it.
Read this whole file before writing code. Everything you need to build and test the client is here or in this folder.

```
client_handoff/
  CLIENT_README.md      this spec
  mock_server.py        dev-only fake API so you can build without the real server
  samples/              three recorded real API responses (small / medium / large trip) + trips.json
```

---

## 1. What this project is (60 seconds)

PathSnap reconstructs the road route a vehicle took from noisy 1 Hz GPS. The **server** (a teammate's laptop, "Laptop 1")
holds the trained model and the road networks and exposes a small HTTP API. **You build the client** ("Laptop 2"): a web UI
that lists the ten evaluation trips, lets an evaluator pick one, calls the API, and shows the reconstructed route and scores.

```
 Laptop 2 (you)                                  Laptop 1 (server teammate)
 ┌───────────────────────┐   GET /health          ┌──────────────────────────────┐
 │ browser UI            │   GET /trips           │ FastAPI + model + road graphs│
 │  trip picker          │ ─────────────────────► │  candidates → classifier →   │
 │  map + results        │   POST /predict        │  HMM/Viterbi → route         │
 │  request log          │ ◄───────────────────── │  scores vs hidden reference  │
 └───────────────────────┘   JSON                 └──────────────────────────────┘
```

Final evaluation: the evaluator opens your UI → sees the ten reserved trips → picks any one → your UI sends the trip id
→ the server runs the full pipeline and scores itself against its private reference data → your UI shows the predicted
route and the results. It must work for **all ten** trips, not one demo trip.

## 2. Hard rules (violating these costs the team marks or disqualifies it)

1. **No ground truth in the client.** The server never sends the reference route and you must not try to obtain it.
   Show only what `/predict` returns.
2. **No hardcoded outputs.** Never embed trip ids, routes, or numbers in the shipped client. The ten trips come from
   `GET /trips` at runtime. Every shown number comes from a live `/predict` response. Never cache a result to replay it.
3. **No bypassing the API.** The client may not read dataset files, run a model, or fake a response on the demo path.
4. **Mock data is dev-only.** `mock_server.py` and `samples/` exist so you can develop offline. If the client ever talks to
   a mock (`/health` returns `"mock": true`, or a response has `"mock": true`), show a **red full-width banner**. Before
   the final hand-in, remove every mock code path from the client.
5. **Independent work.** Code, UI and report are compared across teams. The visual direction below is *inspired by* a
   public website (EDOLUS) but you must write your own code, copy and assets. Do not copy EDOLUS code, wording, wordmark,
   3D models, audio, or images. Do not invent marketing claims, specs or numbers about PathSnap.
6. **AI disclosure.** Every team member must be able to explain the client code in a viva. Keep the code readable and
   commented where non-obvious, and disclose AI assistance as the event requires (see the server repo README).
7. **No secrets** in the repo. **Never commit `node_modules`.**

## 3. Connecting to the server

- The API listens on `http://<SERVER_IP>:8000` (the server teammate gives you the IP; same Wi-Fi/LAN as Laptop 2).
- Default API base in the client: `http://localhost:8000`. Make it overridable with the URL query `?api=http://<SERVER_IP>:8000`
  and with an editable field on the connection-error panel; remember it in `localStorage` (wrap every storage access in try/catch).
- **CORS:** the server allows browser origins `localhost`, `127.0.0.1`, `192.168.x.x`, `10.x.x.x`, `172.16-31.x.x` on any port. If you
  serve the UI from anywhere else, tell the server teammate (they set `PATHSNAP_CORS_REGEX`).
- Quick connectivity test from Laptop 2: `curl http://<SERVER_IP>:8000/health`. If it hangs, the server's Windows firewall
  is blocking port 8000 (their job, see section 10).
- Serve the UI with Vite on the LAN: `npm run dev -- --host` (or `vite build` + `vite preview --host`).

## 4. API contract

All bodies are JSON. Coordinates are **[longitude, latitude]** in degrees (x first). Never use degrees as distances.

### `GET /health`
```json
{"status":"ok","model":"RandomForest (Optuna-tuned) + HMM","n_trips":10,
 "decoder":{"alpha":0.97,"beta":2.97,"invalid_pen":39.1,"gap_s":15.0,"reset_d":500.0}}
```
`mock: true` is present only on the mock server. Show `model` somewhere visible.

### `GET /trips`
```json
[{"trip_id":"00000052","n_gps":262,"duration_s":261.0}, ...]   // exactly the trips the server can serve (ten at evaluation)
```
Use `trip_id` as an opaque string. Sort as returned. Do not assume ids are consecutive or numeric.

### `POST /predict`   body `{"trip_id":"00000052"}`
```jsonc
{
  "trip_id": "00000052",
  "model": "…",
  "gps":     [[lon, lat], …],                       // one per observation, in time order
  "network": [[lon1, lat1, lon2, lat2], …],         // road segments near the trace (undirected, de-duplicated)
  "classifier_only": {
    "arcs":   [[lon1, lat1, lon2, lat2], …],        // segments chosen by the classifier alone, in order (NOT one per observation)
    "breaks": [[lon, lat], …]                       // where that route jumps to a segment that is not connected to the previous one
  },
  "hmm": { "route": [[lon, lat], …] },              // final reconstructed route as one ordered polyline (may be [] if nothing matched)
  "metrics": {
    "scored": true,                                 // false => the server had no reference for this trip; accuracy fields are then absent
    "n_obs": 262,
    "pointwise_acc_classifier": 0.9618, "pointwise_acc_hmm": 0.9924,
    "route_continuity_classifier": 0.916, "route_continuity_hmm": 1.0,
    "route_f1_classifier": 0.860, "route_f1_hmm": 0.920,
    "candidate_miss_rate": 0.0,
    "latency_ms": { "candidates": 40, "classifier": 22, "hmm": 90 }   // server-side stage times; total = sum
  }
}
```
Metric meanings (use these exact definitions in tooltips/labels):
- **Pointwise accuracy**: of the GPS observations that have a reference label, the share whose predicted road segment equals the reference segment. "Classifier only" picks each point's best segment independently; "+ HMM" uses the sequence-aware decoder.
- **Route continuity**: share of transitions between consecutive different predicted segments where the next segment is reachable from the previous within 3 directed segments. 1.0 = no disconnected jumps.
- **Route F1**: overlap (F1) between the set of predicted road segments and the reference set; order ignored.
- **Candidate miss rate**: share of labelled observations whose reference segment was not even among the candidates considered (within 75 m).
- **n_obs**: number of labelled observations the accuracies are computed over (can be slightly below `gps.length`).

Errors: `404 {"detail":"unknown trip_id"}`, `422` malformed body, `500 {"detail":"<ErrorType>: <message>"}`. Network failure = no response.
Treat missing optional fields defensively (`scored:false`, empty `hmm.route`, empty `breaks`).

Performance to design for: `/predict` takes about **0.3 to 3 s** for most trips, and **5 s or more** for the largest ones or when several requests queue (the server handles one at a time); responses are **~25 KB to ~300 KB**
(larger trips can reach a few MB). Use a client timeout of at least 60 s, show a progress state, disable the run button while busy, and never block the UI thread.

## 5. Functional requirements

1. **Connect screen:** on load call `/health` then `/trips`. Show the model name and API address. On failure show a clear error, an editable API URL field and a Retry button. Never crash.
2. **Trip picker:** list every trip from `/trips` (index, id, point count, duration). Select with click or ↑/↓. Enter = run. Works for any number of trips (ten expected).
3. **Run:** `POST /predict` for the selected id. Disable the button while running; show a running state (stage text + elapsed time is fine); handle every error class above with a readable message and let the user retry or pick another trip.
4. **Map** (own canvas rendering): draw the route on top of a real street map when the client laptop has internet (OpenStreetMap tiles under the route, placed with the same projection as the route, attribution "© OpenStreetMap contributors" shown),
   and **fall back automatically to the offline vector road network** (`network`) if no tile loads, so it still works with no internet. Layers: `network` (offline roads), `gps` (small dots), `classifier_only.arcs` (dashed), `classifier_only.breaks` (rings), `hmm.route` (bold, highlighted). Note: CARTO's free tiles now return an "API KEY REQUIRED" watermark; use OpenStreetMap or a keyed provider.
   Layer toggles for each. Pan (drag), zoom (wheel, cursor-anchored), double-click to fit, scale bar, legend, start/end markers on the HMM route.
   Use a local equirectangular projection for drawing: `x = (lon-lon0)·cos(lat0)`, `y = -(lat-lat0)` (lat/lon in degrees → keep aspect right). Must stay smooth on a 4000-point trip (canvas or WebGL, not thousands of DOM nodes).
5. **Results panel:** classifier-only vs +HMM for pointwise accuracy and route continuity (and route F1), the delta, candidate miss rate, number of disconnected jumps (`breaks.length`), n_obs, model name, and the latency breakdown (3 stages + total).
6. **Request log:** a small log of the client's own requests (time, method, path, status, duration). The evaluator is also shown the server console, but this helps.
7. **Mock banner:** see rule 4.
8. **Accessibility basics:** real `<button>`s, visible focus, readable contrast, `prefers-reduced-motion` respected.
9. Works at 1280×720 and up on Chrome/Edge. Mobile is not required.

## 6. Design brief (inspired by the EDOLUS look; original execution)

Mood: dark, cinematic, technical HUD. A flat black canvas, a single warm gold accent, mono-spaced labels, wide-tracked headings.

| Token | Value |
|---|---|
| Background | `#030508` |
| Matched route / accent | `#ffaa33` (secondary `#d49b4b`) |
| Text | white / grey `#c4c7c7`, dim `#707a7d` |
| Disconnected-jump marker | `#ff6a55` |
| Fonts | **JetBrains Mono** (labels, HUD, numbers) + **Space Grotesk** (headings). Bundle via `@fontsource/*`, never a CDN |

Elements: wordmark "PATHSNAP"; boot screen with a percentage counter that tracks the real `/health` → `/trips` steps; character-scramble reveal on headings;
viewfinder corner brackets around the map; custom "+" cursor with faint constellation lines (must not hide the real cursor on buttons/inputs or hurt usability);
sharp rectangular white CTA buttons with mono caps text; hairline panel borders.

**Layout of the app:** left = trip picker + run button; centre = map; right = results + log. Top bar = wordmark, API status dot + address, model name.

**Scroll-driven intro (about 600vh) before the app**, with a persistent **SKIP INTRO** control and a `#app` URL hash (evaluators must reach the picker immediately; remember the choice in `sessionStorage`).
Stages scrubbed by scroll, each with short original copy: (1) raw noisy GPS points, (2) the road network around them, (3) classifier-only route with its disconnected jumps, (4) the HMM/Viterbi route, (5) the real result numbers.
Data for the intro must be **real**: fetch one live `/predict` (default: the trip with the smallest `n_gps`, overridable with `?introTrip=ID`). If the API is unreachable show the text without the map and a notice. Libraries: GSAP + Lenis are fine (bundled, no CDN).

**WebGL post-processing layer on the map canvas only:** subtle radial chromatic aberration (edges), tilt-shift/edge-blur depth-of-field, bloom on the gold route.
Panels and text stay crisp. **Graceful fallback:** no WebGL, or `?fx=0` → plain 2D canvas with a CSS/canvas glow. Keep pan/zoom smooth on the largest trip.

## 7. Suggested stack and structure

Vite + React + TypeScript, no UI framework required, no map-tile library. Keep API access in one module and types in one file.

```
web/
  package.json  vite.config.ts  tsconfig.json  index.html
  src/ main.tsx  App.tsx  api.ts  types.ts  MapCanvas.tsx  Intro.tsx  fx/  styles.css
```

`api.ts` essentials: `getBase()/setBase()`, `health()`, `trips()`, `predict(id)`, a log callback, and a single `fetch` wrapper that times requests.

## 8. Develop without the real server

```
pip install fastapi uvicorn
python mock_server.py            # http://127.0.0.1:8000, serves 3 recorded trips (00000052 small, 00000009 medium, 00000056 large)
```
It serves the same contract with 1 s artificial latency and flags itself as a mock. Test the error paths too (`?api=http://localhost:9999`, stop the mock mid-run, unknown id).
When the real server is reachable on the LAN, switch with `?api=http://<SERVER_IP>:8000` and re-run the whole checklist. The numbers in `samples/` come from an untuned development model and are illustrative only; never quote them.

## 9. Acceptance checklist (run all of it against the REAL server before hand-in)

- [ ] `npm run build` (type-check + bundle) passes with no errors; no console errors on a fresh load.
- [ ] All ten trips run end to end, each rendering a route and metrics; the largest trips stay smooth to pan/zoom.
- [ ] Wrong API URL → readable error + retry; server stopped mid-run → readable error, UI recovers.
- [ ] `/trips` is the only source of the trip list; no ids/results/numbers hardcoded (search the source to confirm).
- [ ] No reference/ground-truth data requested or shown; mock code and `mock-data` files removed from the shipped build.
- [ ] Intro works, can be skipped, and `#app` jumps straight to the picker; `?fx=0` and `prefers-reduced-motion` behave.
- [ ] Works with no internet: fonts are bundled, no CDN; the street-map tiles are optional and the offline vector roads appear automatically when they cannot load.
- [ ] Screenshots captured for the report: connect screen, trip list, a finished run (map + results), and an error state.
- [ ] README for the client (install, run, `?api=` usage) is committed.

## 10. For the human on the server side (Laptop 1)

1. `ipconfig` → note the IPv4 address on the shared network; give it to the client teammate.
2. Allow inbound TCP 8000 (Windows Defender Firewall → Inbound rule → port 8000), or run once in an admin terminal:
   `netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000 profile=any`
   (use `profile=any`: a Wi-Fi network marked "Public" is not covered by a Private-only rule; delete the rule after the event).
3. Point the API at the evaluation trips and start it, binding to all interfaces:
   `set PATHSNAP_TRIPS_DIR=<folder with the ten trips>` then `python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000`.
4. Keep the console visible: it prints one line per request, and the evaluator must be shown it.
5. Verify from the client laptop with `curl http://<SERVER_IP>:8000/health`.

## 11. Hand-in

The client source lives in the repo under `web/` (source + `package.json` + lockfile + README, no `node_modules`, no `dist`).
Send the finished `web/` folder to the server teammate, who commits it to the team's public GitHub repo.
The code shown at evaluation must match the code in the repo.
