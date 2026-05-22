# ValveLens Agent Context Pack

Last updated: 2026-05-09

This document is the fast handoff note for any new contributor or agent working on ValveLens. It is not the full runbook. It is the compact project picture that explains what the system is, what stage it is in, what already exists, and what must happen next.

For fuller detail, read these next:

- `docs/PROJECT_STATUS.md`
- `docs/NEXT_STAGE_V03.md`
- `docs/DEVICE_IDENTITY_BENCHMARK.md`
- `docs/ROBUSTNESS_PREPROCESSING.md`
- `README.md`

## 1. Project identity

- Project name: `ValveLens`
- Current practical stage: `v0.2+`
- Next target stage: `v0.3`
- Main goal: build an uncertainty-aware interactive computer-vision assistant for device localization and identity verification in visually repetitive industrial-like environments
- One-sentence description: ValveLens localizes the scene first, detects candidate devices, tries to identify the correct instance, and asks the user for help when confidence is not strong enough

## 2. Core problem

- industrial facilities contain many visually similar devices
- plain object detection is not enough because it answers only "what object class is here"
- ValveLens must reduce ambiguity by using context first, then identity signals, then an uncertainty-aware decision
- the intended product behavior is not "always predict", but "accept when justified, defer when uncertain"

## 3. System pipeline

Practical runtime pipeline:

```text
Input image / video / webcam frame
-> quality assessment
-> zone retrieval
-> detector inference
-> ROI crop / optional segmentation fallback
-> OCR
-> embedding retrieval for device identity
-> fusion scoring
-> uncertainty policy
-> accept or ask the user
-> observation + feedback logging
```

High-level product intent:

```text
Input image/video
-> Zone localization
-> Device detection
-> Optional OCR/ReID/segmentation
-> Uncertainty decision
-> Human confirmation if needed
```

## 4. Main components

### Backend

- FastAPI service
- modular inference pipeline under `backend/app/`
- detector module
- zone localization / VPR module
- FAISS index helpers
- sqlite metadata store
- OCR, ReID, tracking, fusion, and policy modules

### Frontend

- React + Vite
- webcam, image, and video workflows
- overlay rendering for detections
- right-side panel for zone candidates, decisions, OCR, and device evidence
- feedback actions for confirm, wrong, and tap-select

### Models

- YOLOv8 detector for `valve` / `gauge`
- embedding-based retrieval for zones and devices using local FAISS indices
- OCR path using EasyOCR or pytesseract fallback when available
- segmentation is optional and not the main focus right now

## 5. What has been done

- public-data zone workflow built, mainly around OpenLORIS
- zone keyframe ingestion and FAISS retrieval implemented
- combined YOLO valve/gauge detection dataset prepared
- CUDA YOLO baseline trained and evaluated
- trained detector copied to `models/detector.pt`
- backend detector integration updated to use trained weights by default with robust path resolution
- frontend shows zone candidates, decisions, OCR snippets, and identity evidence
- OCR-first identity logic implemented
- ReID retrieval path implemented
- tracking and temporal smoothing added
- feedback logging improved and session-aware identity carryover helpers added
- metrics export and summary CLIs added
- robustness preprocessing experiment module added for clean/degraded/restored detector comparisons
- identity benchmark workflow added with manifest-based enrollment and validation
- controlled proxy identity benchmark generator added for `V-1023`, `V-2040`, and `PG-45`
- OCR backend diagnostic and OCR smoke-test CLIs added
- project status and next-stage docs added

## 6. Important files and folders

### Backend core

- `backend/app/main.py`
- `backend/app/pipeline.py`
- `backend/app/detector.py`
- `backend/app/zone_localizer.py`
- `backend/app/ocr.py`
- `backend/app/reid.py`
- `backend/app/fusion.py`
- `backend/app/policy.py`
- `backend/app/tracker.py`
- `backend/app/db.py`
- `backend/app/faiss_store.py`
- `backend/app/config.yaml`

### Routes

- `backend/app/routes/infer.py`
- `backend/app/routes/zones.py`
- `backend/app/routes/devices.py`
- `backend/app/routes/feedback.py`
- `backend/app/routes/debug.py`

### Frontend

- `frontend/src/pages/Live.jsx`
- `frontend/src/components/OverlayCanvas.jsx`
- `frontend/src/components/SidePanel.jsx`
- `frontend/src/api.js`

### Models and artifacts

- `models/detector.pt`
- `runs/detect/...`
- `artifacts/detection_training/...`
- `artifacts/robustness/...`
- `artifacts/identity_benchmark/...`
- `data/detection/combined/...`
- `data/device_benchmark/...`

### Docs

- `README.md`
- `docs/PROJECT_STATUS.md`
- `docs/NEXT_STAGE_V03.md`
- `docs/AGENT_CONTEXT_PACK.md`
- `docs/DEVICE_IDENTITY_BENCHMARK.md`
- `docs/ROBUSTNESS_PREPROCESSING.md`

## 7. Research framing

- Working title: `ValveLens: Uncertainty-Aware Interactive Vision for Industrial Device Localization`
- Main claim: contextual zone localization reduces ambiguity before device identification
- Secondary claim: uncertainty should be used as a decision mechanism, not just reported as a metric
- Demonstration style: thesis-friendly prototype using public datasets and simulated device identity rather than real oil-facility deployment

## 8. Current problems and TODO

- proxy device database population has been validated, but real physical device references are still missing
- OCR backend is currently environment-blocked if Tesseract-OCR is not installed or not on PATH
- ReID works on the controlled proxy benchmark, but real industrial identity validation still requires repeated photos of physical devices
- API accepted identity decisions are not yet achieved on tight proxy crops; full-frame proxy scenes or real scenes are likely needed
- confidence is still mostly heuristic rather than calibrated
- UI can explain decisions better
- experiment logging exists but still needs stronger ground-truth discipline
- repo cleanup and documentation should continue as features land

## 9. How things are connected

- zone images -> embeddings -> sqlite + FAISS zone index -> zone prediction
- detection dataset -> YOLO training -> `models/detector.pt` -> backend detector
- detection crops -> OCR / embeddings -> device retrieval -> fusion -> policy
- confidence / ambiguity -> uncertainty action -> accept or defer
- backend API -> frontend visualization -> demo / presentation / experiments
- observations + feedback -> metrics export -> later evaluation

## 10. Current verified state

Known verified items from the recent repo state:

- backend tests pass
- trained detector integration loads from `models/detector.pt`
- detector semantic classes are `valve` and `gauge`
- zone-aware demo path exists
- session-aware feedback helpers exist
- proxy identity benchmark has been generated with:
  - 3 proxy devices
  - 24 reference images
  - 36 query images
- active benchmark DB/index state has been verified with:
  - `devices_count = 3`
  - `device_refs_count = 24`
  - `device_faiss_size = 24`
- ReID is validated on the proxy benchmark:
  - recent top-1/top-k validation around `0.9444`
  - single-image `smoke_reid` for `V-1023` returns top-1 `V-1023`
- OCR diagnostics report `pytesseract` imports, but `tesseract.exe` is missing unless installed locally
- API validation on tight proxy crops has been tested and currently defers all proxy crop queries rather than reaching `ACCEPTED`

Important caveat:

- identity can still appear incomplete in another runtime if `devices_count`, `device_refs_count`, or `device_faiss_size` are zero in `/debug/status`
- proxy identity validation is not the same as final real industrial identity validation

## 11. Instructions for a new agent

- do not restart the project from zero
- preserve the existing architecture unless there is a strong reason to change it
- prefer narrow, high-signal changes over broad redesigns
- keep the system demo-ready, thesis-friendly, and explainable
- before adding new features, check:
  - `docs/PROJECT_STATUS.md`
  - `docs/NEXT_STAGE_V03.md`
  - `backend/app/config.yaml`
  - current `/debug/status`

## 12. Best next step

The next best step is to finish the v0.3 identity checkpoint:

1. install/configure OCR backend, preferably Tesseract-OCR on Windows
2. rerun `check_ocr_backend` and `smoke_ocr` on the easy-tag proxy benchmark
3. generate full-frame proxy scenes or capture real device scenes so the API pipeline sees normal detector/zone evidence
4. validate at least one API or Live session that reaches `ACCEPTED`
5. export metrics and summarize accepted/deferred behavior

That is the shortest path from the current repo state to a true `v0.3` identity-aware assistant.
