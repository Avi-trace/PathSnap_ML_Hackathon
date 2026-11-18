# Demo-day runbook (final evaluation)

Two laptops on the same network. **Laptop 1** = API + model + the ten evaluation trips. **Laptop 2** = the UI. The evaluator picks a trip on Laptop 2;
Laptop 1's console shows the request. Practise this end to end once before the evaluation.

## 0. Before the day

- [ ] Both laptops: repository cloned, `pip install -r requirements.txt` (Laptop 1) and `npm install` in `web/` (Laptop 2) done **while online**.
- [ ] Laptop 1: `artifacts/final/bundle.joblib` is present (it is committed). No training is needed on the day.
- [ ] Decide the network: a **phone hotspot** or a router is safest. Some college/venue Wi-Fi isolates clients (laptops cannot see each other) and the check in step 3 will fail; if so, switch to a hotspot.
- [ ] Laptop 1 firewall: allow inbound TCP 8000 on the profile your network uses (a Wi-Fi marked "Public" is a different profile from "Private").
  Admin terminal: `netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000 profile=any`

## 1. Laptop 1: start the API

Put the ten evaluation trips in one folder, each as `<id>\<id>.track`, `.nodes`, `.arcs` and `.route` (the `.route` files stay on this laptop; they are used only for scoring).

```
conda activate pathsnap
set PATHSNAP_TRIPS_DIR=<folder with the ten evaluation trips>
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

Check the startup lines in the console: model name `RandomForest (Optuna-tuned) + HMM`, the decoder parameters, and `serving : 10 trips from <your folder>`.
If it says a different number of trips or the wrong folder, stop and fix `PATHSNAP_TRIPS_DIR`. Leave this console visible: the evaluator must see it.

Find this laptop's address: `ipconfig` (IPv4 of the active adapter), e.g. `10.8.120.153`.

## 2. Laptop 2: start the UI

```
cd web
npm run dev -- --host          (or: npm run build && npm run preview -- --host)
```

Open `http://localhost:5173/?api=http://<laptop-1-ip>:8000`.

## 3. Verify the link (60 seconds)

- On Laptop 2: `curl http://<laptop-1-ip>:8000/health` returns `{"status":"ok","model":"RandomForest (Optuna-tuned) + HMM", ...}`. No answer = firewall or client isolation.
- In the UI: the boot counter reaches 100 %, the ten trips are listed, the header shows the model name and the Laptop 1 address, no red banner.
- Run one trip: Laptop 1's console prints a `PREDICT trip=... client=<laptop-2-ip> ...` line. **That line is the evidence of a real request** that the brief asks you to show.

## 4. During the evaluation

1. The evaluator opens the UI (use `#app` in the address, or press SKIP INTRO, to go straight to the picker; the scroll intro is optional).
2. The evaluator chooses any of the ten trips and presses RECONSTRUCT ROUTE. Typical response 0.3 to 3.5 s.
3. Point at: the gold HMM route against the classifier-only route and its red jump rings, the accuracy / continuity numbers, and the latency split.
4. Show the Laptop 1 console line for the request. Do not show or discuss the reference route (the UI never receives it).
5. Repeat for other trips; all ten must work.

## 5. If something goes wrong

| Symptom | Fix |
|---|---|
| UI says "cannot reach <address>" | wrong IP in `?api=`, API not running, firewall, or Wi-Fi client isolation. Retry panel lets you edit the address. |
| Browser console shows a CORS error | the UI origin is not a local/private address; start the API with `set PATHSNAP_CORS_REGEX=<regex for your origin>` |
| `404 unknown trip_id` | the UI and API point at different trip folders; restart the API with the right `PATHSNAP_TRIPS_DIR` |
| Map is slow or glitchy | open the page with `?fx=0` |
| A trip takes more than 10 s | very large trip; wait, the API processes one request at a time |
| API crashes | restart the command in step 1; nothing is cached or stored |

## 6. After the evaluation

Screenshot the Laptop 1 console and the UI for the report (section 7 needs the API console, not just the UI).
