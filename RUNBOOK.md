# Demo-day runbook (final evaluation)

Two laptops on the same network. **Laptop 1** = API + model + the ten evaluation trips. **Laptop 2** = the UI. The evaluator picks a trip on Laptop 2;
Laptop 1's console shows the request. Practise this end to end once before the evaluation.

## 0. Before the day

- [ ] Laptop 1: repository cloned, `pip install -r requirements.txt` done **while online**. `artifacts/final/bundle.joblib` is committed, so no training is needed on the day.
- [ ] Laptop 2: either the **client package** (`pathsnap_client_laptop_UI.zip`: only Python 3 needed) or the repository with `npm install` in `web/` done while online.
- [ ] Decide the network. Two setups have been used:
  - **Laptop 2 joins Laptop 1's Windows Mobile Hotspot.** The server's address is then always **`192.168.137.1`**. Laptop 2 gets internet through Laptop 1's connection (needed for the street map).
  - **Both on one router / phone hotspot.** The server's address is Laptop 1's IPv4 from `ipconfig`. Some venue Wi-Fi networks isolate clients (laptops cannot see each other); if the check in step 3 fails, switch to a hotspot.
- [ ] Laptop 1 firewall: allow inbound TCP 8000 on the profile your network uses (a Wi-Fi marked "Public" is a different profile from "Private").
  Admin terminal: `netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000 profile=any`

## 1. Laptop 1: start the API

Put the ten evaluation trips in one folder, each as `<id>\<id>.track`, `.nodes`, `.arcs` and `.route` (the `.route` files stay on this laptop; they are used only for scoring).

**Command Prompt (cmd):**
```
cd /d <path to the pathsnap folder>
set "PATHSNAP_TRIPS_DIR=<folder with the ten evaluation trips>"
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```
**PowerShell** (note: `set` does nothing in PowerShell):
```
cd <path to the pathsnap folder>
$env:PATHSNAP_TRIPS_DIR = "<folder with the ten evaluation trips>"
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

Read the startup lines in the console:
- `model : RandomForest (Optuna-tuned) + HMM` and the decoder parameters;
- **`serving : 10 trips from <your folder>`**. If the number is not 10, or the folder is wrong, stop and fix `PATHSNAP_TRIPS_DIR`. Any folder that was skipped is listed above that line with its reason (for example a missing `.arcs` file);
- a warning `PATHSNAP_TRIPS_DIR is not set: serving the DEVELOPMENT trips` means the variable did not take effect (wrong shell syntax): you would be showing the practice trips.

Leave this console visible: the evaluator must see it. Find this laptop's address with `ipconfig` (IPv4 of the active adapter).

## 2. Laptop 2: start the UI

**Client package (recommended):** unzip, then start it and leave its window open: on Windows double-click `START_CLIENT.bat`; on a Mac run `bash START_CLIENT.sh` in Terminal (or double-click `START_CLIENT.command`; see `README.txt` in the package for the one-time macOS security prompt). Then open in Chrome or Edge:

`http://localhost:8080/?api=http://<laptop-1-address>:8000`   (hotspot setup: `http://localhost:8080/?api=http://192.168.137.1:8000`)

**From the repository:** `cd web`, then `npm run dev -- --host` (or `npm run build && npm run preview -- --host`), and open `http://localhost:5173/?api=http://<laptop-1-address>:8000`.

Open the UI on Laptop 2 itself (`localhost`); the server only accepts browsers on localhost or private-network addresses.

## 3. Verify the link (60 seconds)

- On Laptop 2, open `http://<laptop-1-address>:8000/health` in a browser. You should see `{"status":"ok","model":"RandomForest (Optuna-tuned) + HMM", ...}`. No answer = firewall, wrong address, or client isolation.
- In the UI: the boot counter reaches 100 %, the ten trips are listed, the header shows the model name and the Laptop 1 address, no red banner.
- Run one trip: Laptop 1's console prints a `PREDICT trip=... client=<laptop-2-address> ...` line. **That line is the evidence of a real request** that the brief asks you to show.

## 4. During the evaluation

1. The evaluator opens the UI (add `#app` to the address, or press SKIP INTRO, to go straight to the picker; the scroll intro is optional).
2. The evaluator chooses any of the ten trips and presses RECONSTRUCT ROUTE (or Enter). Most trips answer in about 0.3 to 3 s; the largest can take about 5 s or more on a busy machine, and requests are processed one at a time.
3. Point at: the gold HMM route on the street map, the classifier-only route and its red jump rings, the accuracy / continuity numbers, and the latency split.
4. Show the Laptop 1 console line for the request. Do not show or discuss the reference route (the UI never receives it).
5. Repeat for other trips; all ten must work.

## 5. If something goes wrong

| Symptom | Fix |
|---|---|
| UI says "cannot reach <address>" | wrong address in `?api=`, API not running, firewall, or Wi-Fi client isolation. The retry panel lets you type the right address (with or without `http://`). |
| UI says "no answer from <address> after N s" | the server accepted the connection but did not answer: it is busy with another trip or hung; check Laptop 1's console. |
| Browser console shows a CORS error | the UI is not opened from `localhost` or a private address; open it as `http://localhost:8080/...`, or start the API with `PATHSNAP_CORS_REGEX=<regex for your origin>` |
| `404: unknown trip_id` | the UI and API point at different trip folders; restart the API with the right `PATHSNAP_TRIPS_DIR` (trips added to the folder later appear on the next page load) |
| `422: ...` message under a trip | that trip's input files are unreadable or inconsistent (the message names the file); the other trips are unaffected |
| Map is slow or glitchy | open the page with `?fx=0` |
| No street map under the route (only the route and thin roads) | the client laptop has no internet: the street map is downloaded by the browser of the laptop running the UI. Connect it to a network with internet; the UI says "Base map unavailable" and falls back to the offline road network |
| API will not start: "trips folder does not exist" / "model bundle was not found" | fix the path in `PATHSNAP_TRIPS_DIR` / run from the repository folder; the message says which |
| API crashes | restart the command in step 1; nothing is cached or stored |

## 6. After the evaluation

Screenshot the Laptop 1 console and the UI for the report (section 7 needs the API console, not just the UI).
