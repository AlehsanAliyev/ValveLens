# Next Stage: v0.3 Identity-Aware Assistant

Last updated: 2026-04-12

This document is the next-step runbook for ValveLens. It assumes the current repo already has:

- zone retrieval working
- the trained YOLO detector integrated at `models/detector.pt`
- OCR, ReID, feedback, and session-aware helpers present in code

The goal now is to make identity work end-to-end, not just structurally.

## v0.3 goal

Move ValveLens from:

- a zone-aware detector with logged interaction

to:

- an identity-aware assistant that can:
  - propose real device IDs
  - accept OCR matches when tags are visible
  - retrieve device references from the device FAISS index
  - maintain a confirmed device across nearby frames in the same session
  - export useful experiment metrics

## What must be true before v0.3 is considered real

- `devices_count > 0`
- `device_refs_count > 0`
- `device_faiss_size > 0`
- OCR accepts at least one enrolled printed tag correctly
- ReID returns meaningful top matches for at least one enrolled device
- feedback actions can be tied to a real session and observation history

## Concrete implementation order

### Identity benchmark rule

v0.3 requires enrolled devices and device reference images. Public valve/gauge datasets support detector evaluation, but they do not prove exact device identity unless the same physical device instance appears in reference and query images.

Use `docs/DEVICE_IDENTITY_BENCHMARK.md` for the immediate identity benchmark workflow. The short version is:

- create `data/device_benchmark/devices_manifest.csv`
- create separate reference folders under `data/device_benchmark/refs/<DEVICE_ID>/`
- create held-out query folders under `data/device_benchmark/queries/<DEVICE_ID>/`
- enroll devices with `enroll_devices_from_manifest`
- rebuild the device FAISS index
- validate OCR/ReID/fusion/decision behavior with `validate_identity_benchmark`

This keeps detection evaluation and identity evaluation separate, which is important for a defensible v0.3 claim.

### Step 1: populate the device database

Pick 2 to 5 devices for the first pass. Keep it small and controlled.

Good first examples:

- `V-1023`
- `V-2040`
- `PG-45`

Each device should have:

- one zone assignment
- 3 to 10 reference images
- a clear printed or visible tag if you want OCR to be part of the experiment

Commands:

```powershell
cd d:\python_works\ValveLens\backend
python -m app.cli.create_device --device_id "V-1023" --zone_id "<ZONE_ID>" --type valve --desc "Printed test valve"
python -m app.cli.add_device_refs --device_id "V-1023" --folder "D:\data\devices\V-1023"
python -m app.cli.rebuild_device_index
```

Then verify:

```powershell
curl http://localhost:8000/debug/status
```

Expected direction:

- `devices_count` should be non-zero
- `device_refs_count` should be non-zero
- `device_faiss_size` should be non-zero

### Step 2: verify device retrieval outside the UI

Use the ReID smoke test first. This avoids debugging the full UI path too early.

```powershell
cd d:\python_works\ValveLens\backend
python -m app.cli.smoke_reid --image "D:\data\devices\V-1023\sample.jpg" --topk 5
```

What you want:

- `V-1023` should appear at or near top-1 for its own reference-style image

### Step 3: verify OCR-first acceptance

Prepare one clean image where the printed tag is readable.

Examples:

- `V-1023`
- `PG-45`

Test through the API or the Live page.

What you want:

- OCR text appears in the side panel
- if OCR confidence exceeds `tau_ocr` and the ID is enrolled, the decision becomes `ACCEPTED`

### Step 4: verify live identity output

Use the Live page and inspect:

- zone top-1
- OCR line per detection
- ReID top matches
- fused device output
- final decision

For image mode, start with a still image.
For webcam/video, then test the same object across nearby frames.

### Step 5: verify session-aware behavior

Confirm one device once, then keep the camera on the same object for adjacent frames.

What you want:

- the selected device remains stable more often than before
- later frames benefit from the confirmed device prior rather than dropping immediately back to uncertain

### Step 6: verify experiment export

After one short session, export metrics:

```powershell
cd d:\python_works\ValveLens\backend
python -m app.cli.export_metrics --out data\metrics.csv
python -m app.cli.summarize_metrics --in data\metrics.csv
```

If you have ground truth per session:

```powershell
python -m app.cli.export_metrics --out data\metrics.csv --gt data\gt_sessions.json
```

## Minimal demo checklist for v0.3

Use this as the practical completion checklist.

- zone candidates appear in the side panel
- detector returns `valve` or `gauge` boxes from the trained detector
- OCR reads at least one printed device tag correctly
- ReID returns top device matches for an enrolled device
- one session can reach `ACCEPTED` on a real enrolled device
- feedback rows are written and linked to observations
- metrics export runs after a session

## What to do if it still does not work

### If zone candidates are empty

- check `/debug/status`
- verify `zone_faiss_size > 0`
- confirm the backend was started after index rebuild, or that index reload is happening

### If device matches are empty

- verify `devices_count`, `device_refs_count`, and `device_faiss_size`
- confirm your test image resembles the enrolled reference images
- confirm the crop actually contains the device, not only background

### If OCR is empty

- use a larger, higher-contrast printed tag
- reduce blur and glare
- check that the crop contains the tag text

### If the decision never reaches ACCEPTED

- first test with a still image
- use a device with a clear printed tag
- make sure the device is actually enrolled
- inspect OCR text and ReID top matches in the side panel

## Success criteria for closing v0.3

v0.3 is complete when all of the following are true:

- the backend uses the trained detector by default
- zone retrieval works on the demo dataset
- at least one enrolled device can be accepted by OCR
- at least one enrolled device can be matched through ReID retrieval
- live sessions preserve identity better after confirm/tap-select feedback
- metrics can be exported for a real interaction session

## After v0.3

Only after v0.3 is stable should the project move to:

- stronger evaluation design
- ablations
- calibrated uncertainty
- better device datasets
- any VLM-related work
