# ValveLens Project Status

Last updated: 2026-04-12

This document is the single repo-level status note for ValveLens. Its job is simple: show what exists, what works, what was recently verified, and what is still incomplete so the project does not become hard to navigate.

## Current stage

Current practical stage: `v0.2+`

That means:

- zone retrieval is real and populated
- detector integration is real and now points to the trained YOLO weights
- OCR, ReID, feedback, tracking, and policy are wired into the backend
- identity is only partially operational because enrolled device data is still the main missing ingredient

The next milestone is `v0.3`, which means an identity-aware assistant that can accept a device through OCR or retrieval, keep that identity across nearby frames, and produce meaningful experiment logs.

## What the system currently does

### Backend

- FastAPI app with routes for inference, zones, devices, feedback, and debug status
- sqlite database for zones, keyframes, devices, device references, observations, and feedback
- FAISS indices for zone retrieval and device retrieval
- modular pipeline with:
  - frame quality scoring
  - zone localization
  - detector inference
  - simple tracking
  - ROI crop generation
  - OCR
  - ReID-style embedding retrieval
  - fusion scoring
  - uncertainty policy

### Frontend

- webcam, image, and video workflows
- overlay rendering for detections
- side panel for zone candidates, decision text, OCR, and device evidence
- feedback actions for confirm, wrong, and tap-select
- system status widget using `/debug/status`

### Data and tooling

- OpenLORIS zone import flow
- generic zone and device enrollment CLIs
- smoke tests for zone retrieval and ReID
- metrics export and summarization CLIs
- combined YOLO detection dataset under `data/detection/combined`
- trained detector weights copied to `models/detector.pt`
- expanded multiclass detector experiment under `data/detection/industrial_multiclass`, with separate weights at `models/detector_multiclass.pt`

## What is working now

### Verified working paths

- zone database ingestion from OpenLORIS extracted folders
- zone FAISS rebuild and aggregated zone candidate retrieval
- `/infer/image`, `/infer/video`, and `/infer/webcam/frame` observation creation
- detector integration with `models/detector.pt`
- OCR-first parsing path and device-ID regex extraction
- ReID top-match generation in the pipeline when device references exist
- feedback storage and session-aware identity carryover helpers
- metrics export and summary CLIs

### Recently verified

These were recently rechecked from the codebase and current repo state:

- backend test suite: `10 passed`
- detector backend integration utility loads the trained detector from `models/detector.pt`
- trained detector class names resolve to:
  - `0: valve`
  - `1: gauge`
- second expanded detector experiment exists separately from the runtime baseline:
  - dataset: `data/detection/industrial_multiclass`
  - weights: `models/detector_multiclass.pt`
  - artifacts: `artifacts/detection_multiclass`
  - current test mAP50: `0.4650`
  - current test mAP50-95: `0.3252`
  - status: exploratory thesis/future-product model, not the default backend detector

## What is only partially working

- identity acceptance is implemented in code, but it depends on enrolled devices and reference images
- tap-select and confirm now help the current session more than before, but feedback still does not update the model itself
- OCR works when text is visible and the OCR backend is available, but it is still sensitive to crop quality and tag visibility
- ReID works structurally, but without a populated device index it cannot drive the live assistant meaningfully
- uncertainty policy is real, but some scores are still heuristic rather than calibrated probabilities

## What is still missing

- a populated `devices` and `device_refs` database for the real demo path
- end-to-end identity validation with printed tags and live frames
- a stable evaluation set with known ground-truth device identities
- stronger session/task bookkeeping for more rigorous experiment analysis
- a cleaned-up experiment log or changelog beyond this status note

## Current milestone timeline

This is the short milestone view of what has been added so far. It is not meant to be a full changelog. It is meant to keep the repo mentally organized.

### Milestone 1: v0.1 skeleton

- backend FastAPI service created
- frontend React/Vite app created
- sqlite schema and FAISS storage introduced
- modular pipeline structure established
- initial JSON schemas, policy, and smoke tests added

### Milestone 2: zone-aware system

- zone ingestion CLIs added
- OpenLORIS dataset used as public-data proxy zones
- zone embeddings stored in sqlite
- zone FAISS retrieval implemented
- `/debug/status` added
- frontend started showing zone candidates and decision output

### Milestone 3: dataset and detector work

- combined YOLO detection dataset built from two valve/gauge archives
- dataset inspection and preview scripts added
- CUDA YOLO baseline trained and validated
- best detector weights copied to `models/detector.pt`
- backend detector now resolves model path robustly and uses trained weights by default
- a second oil/gas-oriented multiclass YOLO experiment was trained and evaluated without overwriting the stable valve/gauge baseline

### Milestone 4: identity and interaction improvements

- OCR-first device-ID matching implemented
- ReID retrieval path implemented
- simple tracking and temporal smoothing added
- feedback route improved for tap-select and confirm behavior
- side panel expanded to show OCR and ReID evidence
- session-aware identity carryover helpers added
- metrics export and summarization added

## What has been tested

Current backend tests under `backend/app/tests`:

- `test_schema.py`
- `test_policy.py`
- `test_pipeline_smoke.py`
- `test_tracker.py`
- `test_ocr_identity.py`
- `test_feedback_helpers.py`

Most recent known local result:

```text
10 passed in 0.78s
```

These tests cover:

- schema correctness
- policy decisions
- pipeline smoke behavior
- tracker continuity
- OCR ID parsing and enrolled-ID acceptance logic
- feedback helper behavior

These tests do not yet cover:

- full API integration
- frontend behavior
- FAISS rebuild/reload integration end-to-end
- real detector inference assertions
- experiment-ground-truth evaluation

## What data exists right now

Known project-level data assets:

- OpenLORIS-based zone ingestion pipeline is implemented and used as the public-dataset zone source
- combined detection dataset exists under `data/detection/combined`
- trained detector exists at `models/detector.pt`

Identity status depends on runtime DB contents. If `devices_count` or `device_refs_count` are zero in `/debug/status`, the identity part of the assistant is not populated yet even though the code path exists.

## Recommended working habit

Whenever a meaningful feature lands, update this file in three places:

- `Current stage`
- `Current milestone timeline`
- `What has been tested`

That is enough to keep the repo understandable without turning the README into a long changelog.

## Next document to use

If you are about to continue development, read:

- `docs/NEXT_STAGE_V03.md`

That file is the concrete runbook for the next step toward a true identity-aware assistant.
