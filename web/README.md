# PathSnap web client (Laptop 2)

Evaluator-facing UI: connects to the PathSnap API, lists the evaluation trips, runs the selected trip through the API, and shows the
reconstructed route and scores. Vite + React + TypeScript. Fonts are bundled. The route is drawn on a canvas **on top of an OpenStreetMap street map**
(tiles are fetched by the browser of the machine running the UI, so that machine needs internet for the street map). With no internet the UI says so and shows
the offline vector road network instead, so it still works.

## Run

```
npm install
npm run dev -- --host              # dev server on http://<this-laptop>:5173
# or a production build:
npm run build && npm run preview -- --host
```

Open `http://localhost:5173/?api=http://<server-ip>:8000` (the API address is remembered in `localStorage`; the connection-error panel also has an editable field).
Default API: `http://localhost:8000`.

| URL option | Effect |
|---|---|
| `?api=http://<ip>:8000` | API base address |
| `#app` | skip the scroll intro and go straight to the trip picker |
| `?fx=0` | disable the WebGL post-processing pass (also switches itself off when frames are slow) |
| `?base=0` | do not load the street map (offline road network only) |
| `?tiles=https://.../{z}/{x}/{y}.png` | use another XYZ tile server instead of OpenStreetMap (`{s}` and `{r}` are optional) |
| `?introTrip=<trip_id>` | trip used by the scroll intro (default: the trip with the fewest GPS points) |

## What it shows

Boot screen (tracks the real `/health` then `/trips` calls) -> 600vh scroll intro (five stages, driven by one live `/predict` response) -> app:
trip picker (from `GET /trips`), canvas map with layer toggles (road network, GPS, classifier-only route, HMM route, disconnected jumps), results panel
(classifier-only vs HMM: pointwise accuracy, route continuity, candidate miss rate, jumps, latency) and a request log.

## Rules this client follows

- Every trip, route and number comes from the API at run time: no bundled or cached results. The reference route is never requested or shown (the server keeps it).
- If the API reports `"mock": true` (only the dev mock server in `../client_handoff/mock_server.py` does), a red banner is shown and the numbers must not be quoted.
- All requests are visible in the in-app log; the server prints its own log line for each one.

## Files

`src/api.ts` (all HTTP) · `src/types.ts` (API contract) · `src/App.tsx` (boot, layout, results) · `src/Intro.tsx` (scroll intro, GSAP + Lenis) ·
`src/MapCanvas.tsx`, `src/mapDraw.ts` (2D map) · `src/fx.ts` (WebGL pass) · `src/ErrorBoundary.tsx` · `src/Scramble.tsx`, `src/Cursor.tsx` (HUD details).
