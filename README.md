# ValveLens v0.1

A computer vision guided interactive assistant for device localization and identification.

ValveLens helps an operator find, identify, and verify the exact physical device in visually repetitive environments. It does not try to be a single-shot classifier. It fuses scene context, visual evidence, and uncertainty to decide whether to accept a match or ask for help. The demo is meant for home simulation with public images, printed tags and common household objects.

ValveLens answers a different question than most vision systems. Not just "what is in the image", but "which exact device is this, and how sure are we".

Key ideas are simple and practical. Detection is not identification. Uncertainty is not failure, it is a trigger for interaction. Asking the user can be cheaper than forcing perfect automation. Sparse spatial memory can outperform full mapping in structured facilities.

System overview, conceptual flow:

Camera frame
-> Scene quality assessment (blur, light, glare)
-> Zone localization (visual place recognition)
-> Device detection (candidate objects)
-> ROI refinement (segmentation when available)
-> Identity resolution (OCR and ReID retrieval)
-> Multi-signal fusion and uncertainty estimation
-> Decision policy (accept or ask the user)

This architecture separates four questions. Where am I (zone). What objects exist (detection). Which exact instance is it (identity). Should I trust this result (policy).

## What's included

Backend: a FastAPI service with a modular inference pipeline, FAISS plus sqlite storage, and CLI tools for dataset setup and batch inference. It also includes unit tests for schemas, policy logic, and pipeline sanity.

Frontend: a React (Vite) web interface with live webcam inference, video and image uploads, visual overlays, side-panel decisions, and click or tap based correction.

## Project structure

```
backend/
  app/
    routes/        # infer / zones / devices / feedback
    cli/           # dataset and index management
    tests/         # schema, policy, pipeline checks
  data/            # generated assets (gitignored)

frontend/
  src/
    pages/         # Home / Live / Zones / Devices
    components/    # camera, overlays, panels
```

## Windows PowerShell quickstart

Backend:

```powershell
cd d:\python_works\ValveLens
python -m venv .venv
.\.venv\Scripts\activate
pip install -r backend\requirements.txt
uvicorn app.main:app --reload --port 8000 --app-dir backend
```

Frontend:

```powershell
cd d:\python_works\ValveLens\frontend
npm install
npm run dev
```

Open http://localhost:5173

## Simulating an industrial environment at home

You do not need real oil-field data to demonstrate the system. Create a small but consistent environment and let the system learn it through retrieval.

Step 1, zones. Define 3 to 5 zones such as desk, kitchen, hallway, doorway. Capture 20 to 50 images per zone with varied angles and lighting. These images form the VPR database.

Step 2, devices. Use household objects like knobs or switches, or print images of valves and gauges. Attach simple tags like V-1023 or PG-45. Capture 2 to 10 reference images per device. This is enough to show ambiguity, zone priors, and uncertainty-driven interaction.

## CLI tools

Initialize database:

```powershell
cd backend
python -m app.cli.init_db
```

Create a zone:

```powershell
python -m app.cli.create_zone --name "Zone A" --desc "Kitchen wall"
```

Add zone keyframes:

```powershell
python -m app.cli.add_keyframes --zone_id <ZONE_ID> --folder D:\data\zones\zone_a
```

Rebuild zone index:

```powershell
python -m app.cli.rebuild_zone_index
```

Create a device:

```powershell
python -m app.cli.create_device --device_id "V-1023" --zone_id <ZONE_ID> --type valve --desc "Printed valve tag"
```

Add device references:

```powershell
python -m app.cli.add_device_refs --device_id "V-1023" --folder D:\data\devices\V-1023
```

Rebuild device index:

```powershell
python -m app.cli.rebuild_device_index
```

Batch inference on video:

```powershell
python -m app.cli.run_video --video D:\data\videos\demo.mp4 --out results.json
```

Smoke test (requires a sample image):

```powershell
python -m app.cli.smoke_test --image D:\data\samples\frame.jpg
```

## API overview

```
POST /infer/image
POST /infer/video
POST /infer/webcam/frame
POST /zones/create
POST /zones/{zone_id}/keyframes
POST /zones/rebuild_index
POST /devices/create
POST /devices/{device_id}/refs
POST /devices/rebuild_index
POST /feedback
```

Every inference is stored as an observation, so you get traceability today and a clean path to feedback driven improvements later.

## Make it Work (End-to-End)

This runbook assumes you have a few zone photos and a few device reference photos ready. You can use phone images from your home and printed tags like V-1023 or PG-45 taped next to an object.

Step 1, start the backend:

```powershell
cd d:\python_works\ValveLens
python -m venv .venv
.\.venv\Scripts\activate
pip install -r backend\requirements.txt
uvicorn app.main:app --reload --port 8000 --app-dir backend
```

Step 2, start the frontend:

```powershell
cd d:\python_works\ValveLens\frontend
npm install
npm run dev
```

Step 3, initialize the DB:

```powershell
cd d:\python_works\ValveLens\backend
python -m app.cli.init_db
```

At this point, `GET /debug/status` should show zeros for all counts.

Step 4, create zones and add keyframes:

```powershell
python -m app.cli.create_zone --name "Kitchen" --desc "Sink wall"
python -m app.cli.add_keyframes --zone_id <ZONE_ID> --folder D:\data\zones\kitchen
python -m app.cli.rebuild_zone_index
```

Now `/debug/status` should show non-zero `zones`, `zone_keyframes`, and `faiss.zones`.

Step 5, create a device and add references:

```powershell
python -m app.cli.create_device --device_id "V-1023" --zone_id <ZONE_ID> --type valve --desc "Printed tag"
python -m app.cli.add_device_refs --device_id "V-1023" --folder D:\data\devices\V-1023
python -m app.cli.rebuild_device_index
```

Now `/debug/status` should show non-zero `devices`, `device_refs`, and `faiss.devices`.

Step 6, test webcam inference:

Open `http://localhost:5173`, go to Live, select Webcam, and watch the decision updates. If OCR sees the tag and the device exists, the decision should move to ACCEPTED.

Step 7, test tap-select feedback:

Click "Tap Select" and click a box. The decision should update to ACCEPTED with the selected device in the UI.

Step 8, run a smoke test from the CLI:

```powershell
python -m app.cli.smoke_test --image D:\data\samples\frame.jpg
```

### Troubleshooting

If zone candidates are empty, ensure you added keyframes and rebuilt the zone index. Check `/debug/status` for `zone_keyframes` and `faiss.zones` counts.

If device matches are empty, ensure device refs are added and the device index is rebuilt. Verify `device_refs` and `faiss.devices` in `/debug/status`.

If OCR is empty, make sure the tag text is large and high contrast, and that EasyOCR or Tesseract is installed. The system runs without OCR, but it will rely on ReID and policy instead.

If FAISS size is 0, rebuild the index after adding data. The index is not auto built.

If the frontend does not update after tap select, check the `/feedback` response in the browser devtools. The response should include a `decision` object for tap_select, confirm, and reject.

## Configuration

Edit backend/app/config.yaml:

```
tau_zone: 0.65
tau_det: 0.40
tau_ocr: 0.70
tau_reid: 0.50
tau_gap: 0.08
tau_blur: 0.60
tau_low_light: 0.35
frame_stride: 5
```

These thresholds control when the system accepts and when it asks.

## Model dependencies

OpenCLIP ViT-B/32 provides image embeddings for zones and devices. YOLO (ultralytics) handles detection. EasyOCR reads tags. All components have safe fallbacks so the demo runs end to end even without GPUs.

Optional segmentation is supported via Segment Anything. Set SAM_CHECKPOINT to a local checkpoint path to enable mask refinement.

## Tests

```powershell
cd backend
pytest app\tests
```

## Roadmap beyond v0.1

Next steps include stronger industrial ReID embeddings, vision language grounding for natural queries, calibrated uncertainty estimation, AR overlays, and edge device deployment. The core idea remains the same: ValveLens is not a black box, it is a collaborative assistant that knows when to ask for help.
