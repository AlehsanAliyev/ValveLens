# ValveLens Project Status

Last updated: 2026-05-22

This document is the single repo-level status note for ValveLens. Its job is simple: show what exists, what works, what was recently verified, and what is still incomplete so the project does not become hard to navigate.

## Current stage

Current practical stage: `v0.5.1 final thesis/demo package prepared`

That means:

- zone retrieval is real and populated
- detector integration is real and now points to the trained YOLO weights
- OCR, ReID, feedback, tracking, and policy are wired into the backend
- the device inventory is populated for a controlled oil/gas proxy benchmark
- device FAISS/ReID is populated and validated on proxy reference/query images
- OCR is available through Tesseract on Windows, but OCR accuracy is still condition-sensitive
- backend API identity acceptance has been verified on the proxy benchmark
- evidence-aware `/ask` interaction is implemented with rule-based answers and VLM fallback scaffolding
- image inference audit tooling now diagnoses detector, quality, OCR, ReID, fusion, and policy causes for weak demo outputs
- class-name display now preserves detector model names, so expanded classes such as `pipe`, `flange`, or `tank` do not silently become `unknown`
- ReID candidates are grouped by unique `device_id` before ambiguity checks, so repeated references for one device do not create false ambiguity
- VLM visual-understanding mode is implemented behind config/env gates and remains disabled by default
- v0.5 assistant demo artifacts have been regenerated from accepted and uncertain observations
- final thesis/demo result summaries are collected under `artifacts/final_results`

The v0.3 milestone is closed for a controlled proxy benchmark. v0.5 adds an interactive assistant layer that answers from ValveLens evidence instead of blind image guessing. v0.5.1 packages the final thesis/demo outputs without adding datasets, retraining YOLO, replacing `models/detector.pt`, integrating runtime preprocessing, or hardcoding API keys.

## What the system currently does

### Backend

- FastAPI app with routes for inference, zones, devices, feedback, and debug status
- evidence-aware `/ask` route for interactive assistant questions
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
- grouped ReID display and bullet decision reasons
- Assistant / Visual Understanding card with Ask and Describe Image actions
- local demo sample picker for reliable upload examples
- feedback actions for confirm, wrong, and tap-select
- question input and answer panel for evidence-aware assistant responses
- system status widget using `/debug/status`

### Data and tooling

- OpenLORIS zone import flow
- generic zone and device enrollment CLIs
- smoke tests for zone retrieval and ReID
- metrics export and summarization CLIs
- combined YOLO detection dataset under `data/detection/combined`
- trained detector weights copied to `models/detector.pt`
- expanded multiclass detector experiment under `data/detection/industrial_multiclass`, with separate weights at `models/detector_multiclass.pt`
- oil/gas proxy inventory benchmark under `data/device_benchmark`

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
- controlled proxy identity benchmark enrollment, device FAISS rebuild, ReID validation, OCR validation, and API decision validation
- structured evidence extraction and rule-based `/ask` answers for common operator questions
- audit CLI: `python -m app.cli.audit_inference_image --image PATH --model models\detector.pt --also-model models\detector_multiclass.pt`
- VLM diagnostics: `python -m app.cli.check_vlm_backend`
- VLM smoke fallback: `python -m app.cli.smoke_vlm_assistant --image PATH --question "What do you see in this image?" --use-vlm`

### Recently verified

These were recently rechecked from the codebase and current repo state:

- backend test suite: `18 passed`
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
- v0.3 controlled proxy identity benchmark:
  - devices: `11`
  - device references: `184`
  - device FAISS size: `184`
  - query images: `120`
  - missing query files: `0`
  - ReID top-1 accuracy: `0.9917`
  - ReID top-k accuracy: `0.9917`
  - OCR backend: `tesseract`
  - OCR visible-tag exact matches: `67/85` (`0.7882`)
  - API evaluated images: `120`
  - API accepted/deferred: `37/83`
  - API errors: `0`
  - at least one accepted decision: `true`
  - evidence: `artifacts/identity_benchmark/identity_benchmark_summary.json`
- v0.3 metrics export:
  - exported task rows: `63`
  - output: `backend/data/metrics_v03.csv`
  - summary artifact: `artifacts/v03_demo/v03_identity_validation_report.md`
- v0.5 interactive assistant scaffold:
  - evidence module: `backend/app/evidence.py`
  - ask route: `backend/app/routes/ask.py`
  - VLM scaffold: `backend/app/vlm_assistant.py`
  - demo CLI: `backend/app/cli/demo_assistant_queries.py`
  - frontend question UI: `frontend/src/components/SidePanel.jsx`
  - mode: rule-based by default, VLM disabled in config
- backend tests: `18 passed`
- frontend build: passed
  - demo artifacts: `artifacts/v05_assistant_demo`
  - final summaries: `artifacts/final_results`

## What is only partially working

- identity acceptance is implemented and verified on a controlled proxy benchmark, but not yet validated with real repeated physical device photos
- tap-select and confirm now help the current session more than before, but feedback still does not update the model itself
- OCR works when text is visible and the OCR backend is available, but it is still sensitive to crop quality and tag visibility
- ReID works with the populated proxy device index; real deployment confidence still depends on collecting real reference/query images
- uncertainty policy is real, but some scores are still heuristic rather than calibrated probabilities
- the first generated full-frame proxy scenes are useful as preview artifacts, but they did not yet produce accepted decisions because detector/quality evidence was weaker than tight crop benchmark evidence
- VLM provider execution is scaffolded but disabled by default; the current assistant uses deterministic evidence rules

## What is still missing

- repeated real-device reference and query images for external identity validation
- end-to-end identity validation on real full-frame industrial-style scenes
- a stable evaluation set with known ground-truth device identities
- stronger session/task bookkeeping for more rigorous experiment analysis
- a cleaned-up experiment log or changelog beyond this status note
- real VLM provider execution after credentials/model configuration, with tests that it does not invent unsupported device IDs

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

### Milestone 5: v0.3 proxy identity closure

- oil/gas proxy inventory benchmark generated from prepared industrial object crops
- device manifest and query manifest populated under `data/device_benchmark`
- 11 proxy devices enrolled with 184 reference images
- device FAISS rebuilt with 184 vectors
- OCR backend diagnosed and enabled through Tesseract
- API validation run on 120 proxy query images with 37 accepted identity decisions
- v0.3 closure artifacts saved under `artifacts/identity_benchmark` and `artifacts/v03_demo`

### Milestone 6: v0.5 interactive assistant scaffold

- structured evidence object added for observations, detections, OCR, ReID, fusion, decision, quality, and feedback
- `/ask` route added for questions such as `What is this?`, `Where is V-1023?`, and `Why are you uncertain?`
- rule-based answers implemented first for reliability and testing
- VLM assistant scaffold added, but provider execution remains disabled by default
- Live frontend side panel now shows answer mode, evidence chips, recommended next action, and uncertainty reason
- scripted assistant demo validates accepted and uncertain observations through `/ask`

### Milestone 7: v0.5.1 final thesis/demo package

- assistant demo CLI aligned with the final question set:
  - `What is this?`
  - `Where is V-1023?`
  - `Which devices are visible?`
  - `Why are you uncertain?`
  - `What should I do next?`
  - `What tag did you read?`
- assistant artifacts regenerated under `artifacts/v05_assistant_demo`
- thesis assistant section expanded with rule-based and VLM-gated design notes
- final result summaries collected under `artifacts/final_results`
- README and v0.5 docs updated for final claim boundaries

### Milestone 8: demo explanation hardening

- detector outputs now include `class_id`, raw `class_name`, confidence, and bbox through the backend schema
- overlay labels use detector class names rather than a collapsed semantic fallback
- policy decisions now carry explicit `reasons` and `next_action`
- quality text names blur only when `is_blurry` is true from the blur metric
- ReID top matches are grouped by device before computing identity gaps
- `/demo/samples` and `/demo/infer_sample` provide a local sample picker for stable demos
- `/ask` can call a gated VLM provider with image input when configured; otherwise it reports a rule-based fallback reason

For reliable demos, prefer:

- `data/device_benchmark/queries/`
- `data/device_benchmark/manual_v1023/queries/`
- `data/device_benchmark/fullframe_demo/`
- `data/detection/combined/test/images/`
- `data/detection/industrial_multiclass/test/images/`

## What has been tested

Current backend tests under `backend/app/tests`:

- `test_schema.py`
- `test_policy.py`
- `test_pipeline_smoke.py`
- `test_tracker.py`
- `test_ocr_identity.py`
- `test_feedback_helpers.py`
- `test_evidence.py`
- `test_ask_route.py`

Most recent known local result:

```text
18 passed in 15.29s
```

These tests cover:

- schema correctness
- policy decisions
- pipeline smoke behavior
- tracker continuity
- OCR ID parsing and enrolled-ID acceptance logic
- feedback helper behavior
- evidence object creation
- rule-based ask answers
- VLM-unavailable fallback behavior

These tests do not yet cover:

- full browser interaction beyond production build
- FAISS rebuild/reload integration end-to-end
- real detector inference assertions
- experiment-ground-truth evaluation

## What data exists right now

Known project-level data assets:

- OpenLORIS-based zone ingestion pipeline is implemented and used as the public-dataset zone source
- combined detection dataset exists under `data/detection/combined`
- trained detector exists at `models/detector.pt`
- controlled oil/gas proxy identity benchmark exists under `data/device_benchmark`
- v0.3 identity benchmark outputs exist under `artifacts/identity_benchmark`
- v0.3 closure report exists under `artifacts/v03_demo`
- interactive assistant documentation exists at `docs/INTERACTIVE_ASSISTANT.md`

Identity status now has a controlled proxy validation path. If `/debug/status` shows `devices_count`, `device_refs_count`, or `device_faiss_size` as zero in a fresh environment, rerun the manifest enrollment and device FAISS rebuild before testing identity.

## Recommended working habit

Whenever a meaningful feature lands, update this file in three places:

- `Current stage`
- `Current milestone timeline`
- `What has been tested`

That is enough to keep the repo understandable without turning the README into a long changelog.

## Next document to use

If you are about to continue identity validation, read:

- `docs/NEXT_STAGE_V03.md`

If you are about to continue interactive assistant work, read:

- `docs/NEXT_STAGE_V05.md`

For final thesis/demo evidence, start with:

- `artifacts/final_results/README.md`
