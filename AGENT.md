# Project Context Pack

## 1. Project Identity

- name: `ValveLens`
- current stage/version: `v0.2+`, moving toward `v0.3`
- goal: build an uncertainty-aware interactive vision assistant for zone localization, device detection, and device identity verification in visually repetitive industrial-like environments
- one-sentence description: ValveLens localizes context first, detects candidate devices, attempts identity resolution, and defers to the user when confidence is not sufficient

## 2. Core Problem

- industrial environments contain many visually similar valves, gauges, and panels
- plain detection answers object class, not exact instance identity
- the system must reduce ambiguity before identity resolution
- zone/context is used as a prior before OCR and embedding retrieval
- the product must avoid forced guesses and ask for interaction when uncertainty is high

## 3. System Pipeline

- practical runtime pipeline:
  - input image / video frame / webcam frame
  - frame quality scoring
  - zone retrieval from embeddings + FAISS
  - detector inference
  - ROI crop generation with optional segmentation fallback
  - OCR on crop
  - embedding retrieval for device identity
  - tracking and temporal smoothing for video/webcam
  - fusion scoring
  - uncertainty policy
  - response JSON build
  - observation logging
  - feedback logging and session-aware carryover helpers
- high-level product pipeline:
  - input
  - zone localization
  - device detection
  - optional OCR / ReID / segmentation
  - uncertainty decision
  - accept or ask the user

## 4. Main Components

### Backend

- FastAPI app under `backend/app/`
- inference pipeline orchestration in `backend/app/pipeline.py`
- detector module in `backend/app/detector.py`
- zone retrieval in `backend/app/zone_localizer.py`
- OCR, ReID, fusion, policy, tracker modules
- sqlite persistence in `backend/app/db.py`
- FAISS storage helpers in `backend/app/faiss_store.py`
- routes for infer, zones, devices, feedback, debug

### Frontend

- React + Vite app under `frontend/`
- `Live.jsx` is the main demo surface
- overlay rendering in `OverlayCanvas.jsx`
- decision and evidence display in `SidePanel.jsx`
- webcam, image, and video workflows are present

### Models

- YOLOv8 detector with trained weights at `models/detector.pt`
- embedding-based retrieval for zones and devices via FAISS
- OCR path using EasyOCR when available, pytesseract fallback otherwise
- segmentation is optional and not a current priority

## 5. What Has Been Done

- modular FastAPI + React application skeleton built
- sqlite schema and FAISS storage implemented
- OpenLORIS-based public-data zone workflow implemented
- zone ingestion, embedding storage, and zone FAISS retrieval implemented
- `/debug/status` added
- combined YOLO valve/gauge detection dataset prepared
- CUDA YOLO baseline trained and evaluated
- trained detector copied to `models/detector.pt`
- backend detector updated to use trained weights by default with robust path resolution
- image, video, and webcam inference routes implemented
- OCR-first device-ID parsing and matching implemented
- device embedding retrieval path implemented
- tracking and temporal smoothing implemented
- feedback capture improved with session-aware identity carryover helpers
- metrics export and summarization CLIs added
- robustness preprocessing experiment module added for degraded/restored detector evaluation
- device identity benchmark workflow added with manifest-based enrollment and validation
- proxy device benchmark generator added for `V-1023`, `V-2040`, and `PG-45`
- OCR backend diagnostic and OCR smoke-test CLIs added
- project status, next-stage, and agent-context docs added

## 6. Important Files and Folders

- backend:
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
  - `backend/app/routes/`
  - `backend/app/cli/`
  - `backend/app/tests/`
- frontend:
  - `frontend/src/pages/Live.jsx`
  - `frontend/src/components/OverlayCanvas.jsx`
  - `frontend/src/components/SidePanel.jsx`
  - `frontend/src/api.js`
- models and artifacts:
  - `models/detector.pt`
  - `data/detection/combined/`
  - `data/device_benchmark/`
  - `runs/detect/`
  - `artifacts/detection_training/`
  - `artifacts/robustness/`
  - `artifacts/identity_benchmark/`
- docs:
  - `README.md`
  - `docs/PROJECT_STATUS.md`
  - `docs/NEXT_STAGE_V03.md`
  - `docs/AGENT_CONTEXT_PACK.md`
  - `docs/DEVICE_IDENTITY_BENCHMARK.md`
  - `docs/ROBUSTNESS_PREPROCESSING.md`
  - `AGENT.md`

## 7. Research Framing

- title: `ValveLens: Uncertainty-Aware Interactive Vision for Industrial Device Localization`
- main claim: zone/context localization reduces ambiguity before device identification
- secondary claim: uncertainty is used as a decision mechanism, not only as a reported score
- demo positioning: thesis-oriented prototype using public datasets and simulated device identity, not a real oil-facility deployment

## 8. Current Problems / TODO

- proxy device database and device FAISS have been populated and validated, but real physical device references are still missing
- OCR is environment-blocked when Tesseract-OCR is not installed or not on PATH
- ReID is validated on the controlled proxy benchmark, but not yet on real repeated physical device images
- API accepted identity decision has not yet been achieved on tight proxy crops
- uncertainty scores remain mostly heuristic rather than calibrated
- frontend explanation of decisions can still be clearer
- evaluation with ground-truth device identity is incomplete
- environment consistency is fragile when backend and training use different Python installations
- repo still contains legacy or mixed dataset paths and artifacts

## 9. How Things Are Connected

- zone images -> embeddings -> sqlite + FAISS zone index -> zone candidates in inference
- detection dataset -> YOLO training -> `models/detector.pt` -> backend detector
- detection crops -> OCR and embedding retrieval -> fused identity evidence
- fusion outputs + quality + retrieval scores -> policy action
- backend API responses -> frontend overlays and side panel
- observations + feedback -> metrics export and later evaluation

## 10. Verified State

- backend tests pass
- trained detector integration resolves and loads `models/detector.pt`
- detector semantic classes are `valve` and `gauge`
- zone-aware demo path exists and returns zone candidates
- detector is integrated into backend inference by default
- session-aware feedback helpers exist
- metrics export and summary CLIs exist
- proxy identity benchmark generated:
  - 3 proxy devices
  - 24 reference images
  - 36 query images
- active benchmark identity state verified:
  - `devices_count = 3`
  - `device_refs_count = 24`
  - `device_faiss_size = 24`
- proxy ReID is working:
  - recent manifest validation around `0.9444` top-1/top-k
  - `smoke_reid` for `V-1023` returns top-1 `V-1023`
- OCR diagnostic currently shows `pytesseract` imports but `tesseract.exe` is missing unless installed locally
- API validation on tight proxy crops currently defers all queries instead of reaching `ACCEPTED`

## 10A. Latest Current State vs Historical State

- historical state: identity was mostly structural because device references were empty
- current state: the empty-reference blocker is fixed for the controlled proxy benchmark
- remaining blocker: OCR backend availability and at least one accepted API/Live identity path
- important limitation: proxy identity validates the mechanics of OCR/ReID/fusion/decision logging, not final real industrial identity accuracy

## 11. Instructions for New Agent

- do not restart the project from zero
- do not redesign architecture without a specific technical reason
- preserve the current backend/frontend split
- prioritize demo readiness, traceability, and explainability
- check current docs before coding:
  - `docs/PROJECT_STATUS.md`
  - `docs/NEXT_STAGE_V03.md`
  - `README.md`
- check runtime state before assuming missing functionality:
  - `/debug/status`
  - current sqlite contents
  - current FAISS index sizes
- prefer narrow, testable changes
- do not break stable detector or zone-retrieval paths while working on identity features

## 12. Next Best Step

- install or configure OCR backend on Windows, preferably Tesseract-OCR
- rerun `python -m app.cli.check_ocr_backend`
- rerun `python -m app.cli.smoke_ocr --image "..\data\device_benchmark\queries\V-1023\clean\q001.jpg" --expected V-1023`
- generate full-frame proxy scenes or capture real device scenes so the runtime API sees normal detector/zone evidence
- run one complete API or Live session that reaches `ACCEPTED`
- export metrics from that session and inspect accepted/deferred behavior

## 13. Key Decisions (WHY)

- zone-first architecture:
  - chosen to reduce device ambiguity before identity resolution
- YOLOv8 detector:
  - chosen as the practical baseline for training and local deployment
- trained detector stored at `models/detector.pt`:
  - chosen so backend uses one stable model artifact by default
- FAISS + sqlite split:
  - sqlite is source of truth, FAISS is retrieval acceleration
- OCR-first identity logic:
  - chosen because visible device tags can short-circuit retrieval ambiguity
- fallback behavior in detector and OCR:
  - kept to preserve demo continuity during missing dependency cases
- public-data zone workflow:
  - chosen because real industrial facility data is not available
- session-aware feedback helpers:
  - added so identity can persist across nearby frames before any full learning loop exists

## 14. Known Weak Points

- identity path is data-dependent and may appear broken when devices are not enrolled
- confidence thresholds are heuristic and not calibrated
- OCR quality depends heavily on crop quality, blur, glare, and printed-tag visibility
- OCR execution also depends on a working local OCR backend, not just the Python package
- ReID uses embedding retrieval, not a dedicated industrial instance-ID model
- current proxy benchmark uses generated crops and synthetic tags; real device photos are still needed for stronger validation
- fallback paths can mask environment issues if logs are ignored
- mixed Python environments can cause backend/runtime inconsistency
- some documentation and legacy dataset paths may drift as the repo evolves

## 15. Stable Parts (DO NOT TOUCH)

- detector response schema expected by the pipeline
- zone retrieval flow based on sqlite embeddings + FAISS search
- trained detector artifact path: `models/detector.pt`
- core inference route structure:
  - `/infer/image`
  - `/infer/video`
  - `/infer/webcam/frame`
- observation and feedback persistence model
- frontend Live page as the main demo surface
- current docs structure:
  - `README.md`
  - `docs/PROJECT_STATUS.md`
  - `docs/NEXT_STAGE_V03.md`
  - `docs/AGENT_CONTEXT_PACK.md`
