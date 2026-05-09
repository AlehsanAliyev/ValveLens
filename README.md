# ValveLens

ValveLens is an uncertainty-aware industrial vision assistant for finding, detecting, and identifying devices in visually repetitive facility-like environments.

The project is built around a simple research position:

> Industrial perception is not solved by asking "what object is in this image?" The operational question is usually "where am I, which exact device is this, how reliable is the evidence, and what should happen when the system is not sure?"

ValveLens therefore treats perception as a composed evidence pipeline rather than a single detector call. It combines scene context, object detection, OCR, retrieval-based identity, quality diagnostics, score fusion, and an uncertainty policy. When evidence is strong, the system can accept an identity. When evidence is weak, it defers and asks for user interaction instead of pretending to know.

Current practical stage: `v0.2+`, moving toward `v0.3 identity-aware assistant`.

## Current Status

The repository is no longer a `v0.1` skeleton. The current implementation includes:

- FastAPI backend with image, video, webcam, feedback, debug, devices, zones, and ask routes
- React/Vite frontend with live image/webcam/video workflows, overlays, side-panel evidence, feedback, and a small rule-based question interface
- trained YOLOv8 valve/gauge detector at `models/detector.pt`
- OpenCLIP embeddings plus FAISS for zone retrieval and device ReID-style retrieval
- SQLite metadata store for zones, keyframes, devices, references, observations, and feedback
- OCR device-ID parsing with EasyOCR or pytesseract fallback
- tracker and session-aware identity carryover helpers
- score fusion and uncertainty policy
- metrics export and summarization tools
- robustness preprocessing experiment module
- manifest-based device identity benchmark workflow
- proxy device identity benchmark generated from the detector dataset
- structured evidence layer for future VLM-assisted explanations

Most recent local benchmark state:

| Area | Current state |
| --- | --- |
| Detector | Trained YOLOv8 model integrated at `models/detector.pt` |
| Detector classes | `valve`, `gauge` |
| Original detector mAP50 | about `0.870` in the latest robustness artifact |
| Proxy devices | `3` devices: `V-1023`, `V-2040`, `PG-45` |
| Proxy refs | `24` reference images |
| Proxy queries | `36` query images |
| Device FAISS | `24` indexed device references |
| Proxy ReID | latest manifest validation about `0.944` top-1/top-k |
| ReID smoke test | `V-1023` returns top-1 `V-1023`, score about `0.991` |
| OCR | environment-blocked if `tesseract.exe` is missing or EasyOCR is unavailable |
| API acceptance | current tight proxy crops defer all API queries; full-frame proxy scenes or real scenes are still needed |
| Backend tests | `10 passed` in the latest local run |

Important: the proxy benchmark validates the identity pipeline mechanically. It does not prove final real industrial identity accuracy. Real repeated images of physical devices are still needed for stronger external validation.

## Why ValveLens Exists

Industrial facilities create a different visual problem from ordinary object recognition:

- many valves and gauges look nearly identical
- device identity depends on local context, tags, and layout, not only visual category
- lighting, glare, blur, occlusion, dirt, and metallic reflections are common
- public datasets rarely contain zones, exact device instance IDs, OCR tags, and adverse conditions together
- a wrong confident answer can be worse than asking the operator to confirm

ValveLens decomposes the problem into smaller testable questions:

| Question | Module | Output |
| --- | --- | --- |
| Where is this frame likely from? | Zone retrieval / VPR | zone candidates |
| What physical objects are visible? | YOLOv8 detector | valve/gauge boxes |
| Is there a readable ID tag? | OCR | parsed device IDs |
| Does the crop look like an enrolled device? | embedding retrieval / FAISS | ReID candidates |
| How strong is the combined evidence? | fusion | device score and breakdown |
| Should the system accept or ask? | uncertainty policy | `ACCEPTED` or `UNCERTAIN` |

This makes the system easier to evaluate and easier to defend. Each subsystem can be tested independently, then integrated into an end-to-end assistant.

## System Diagram

```mermaid
flowchart TD
    A[Input image, video frame, or webcam frame] --> Q[Image quality diagnostics]
    A --> Z[Zone localization / visual place retrieval]
    A --> D[YOLOv8 valve/gauge detection]

    Z --> ZC[Zone candidates]
    D --> ROI[Detection ROIs]
    ROI --> OCR[OCR device ID parsing]
    ROI --> RID[Device ReID embedding retrieval]

    OCR --> F[Fusion scoring]
    RID --> F
    ZC --> F
    Q --> P[Uncertainty policy]
    F --> P

    P -->|strong evidence| ACC[ACCEPTED identity]
    P -->|weak or ambiguous evidence| ASK[ASK_TAP / defer to user]

    ACC --> LOG[Observation and metrics logging]
    ASK --> FB[Feedback / tap select / confirm]
    FB --> LOG
```

## Research Framing

ValveLens is best understood as a research and engineering prototype, not a production oil-facility deployment.

The working research framing is:

- Context reduces ambiguity before identity resolution.
- Detection is not identity.
- OCR and retrieval are complementary identity signals.
- Uncertainty should control the system's behavior, not just appear as a number in logs.
- User interaction is part of the perception loop when visual evidence is insufficient.
- Robustness should be measured before runtime preprocessing is added.

### Main Contribution

ValveLens contributes a modular, uncertainty-aware architecture for industrial-like device localization and identity verification. It shows how zone retrieval, detection, OCR, ReID, fusion, quality assessment, feedback, and experiment logging can be composed into a transparent assistant.

### What This Project Does Not Claim Yet

ValveLens does not currently claim:

- final industrial deployment readiness
- exact real-device identity accuracy on a real facility dataset
- calibrated uncertainty probabilities
- that preprocessing should already be enabled in runtime inference
- that a VLM should directly replace the perception pipeline

It is fair to claim:

- detector training and evaluation exist for valve/gauge candidates
- zone retrieval exists using public indoor-place datasets
- the identity path is implemented and mechanically validated on a controlled proxy benchmark
- ReID works on the generated proxy identity benchmark
- OCR is blocked by local OCR backend availability, not by missing project architecture
- the API can report uncertainty and defer instead of forcing identity
- robustness preprocessing experiments exist as separate evidence, not runtime behavior

## Architecture

```mermaid
flowchart LR
    subgraph Frontend[React Frontend]
        Live[Live.jsx]
        Canvas[OverlayCanvas.jsx]
        Panel[SidePanel.jsx]
        API[api.js]
    end

    subgraph Backend[FastAPI Backend]
        Infer[/infer routes/]
        Ask[/ask route/]
        Feedback[/feedback route/]
        Debug[/debug/status/]
        Pipeline[pipeline.py]
        Evidence[evidence.py]
    end

    subgraph Perception[Perception Modules]
        Quality[quality.py]
        Zone[zone_localizer.py]
        Detector[detector.py]
        OCR[ocr.py]
        ReID[reid.py]
        Fusion[fusion.py]
        Policy[policy.py]
        Tracker[tracker.py]
    end

    subgraph Storage[Storage]
        SQLite[(SQLite)]
        ZoneFAISS[(FAISS zones)]
        DeviceFAISS[(FAISS devices)]
        Models[models/detector.pt]
    end

    Live --> API --> Infer --> Pipeline
    Pipeline --> Quality
    Pipeline --> Zone
    Pipeline --> Detector
    Pipeline --> OCR
    Pipeline --> ReID
    Pipeline --> Fusion
    Pipeline --> Policy
    Pipeline --> Tracker
    Pipeline --> SQLite
    Zone --> ZoneFAISS
    ReID --> DeviceFAISS
    Detector --> Models
    Ask --> Evidence --> SQLite
    Feedback --> SQLite
    Debug --> SQLite
    Debug --> ZoneFAISS
    Debug --> DeviceFAISS
    Pipeline --> API --> Canvas
    Pipeline --> API --> Panel
```

## Runtime Pipeline

The backend pipeline is implemented in `backend/app/pipeline.py`.

```mermaid
sequenceDiagram
    participant U as User / Camera
    participant API as FastAPI infer route
    participant P as Pipeline
    participant Z as Zone FAISS
    participant Y as YOLOv8 Detector
    participant O as OCR
    participant R as Device FAISS
    participant F as Fusion
    participant C as Policy
    participant DB as SQLite

    U->>API: image / frame / video
    API->>P: run inference
    P->>P: compute blur, brightness, glare
    P->>Z: retrieve zone candidates
    Z-->>P: top-k zones
    P->>Y: detect valves/gauges
    Y-->>P: boxes and confidences
    P->>O: read crop text
    O-->>P: raw OCR text, confidence, parsed IDs
    P->>R: search device embeddings
    R-->>P: device candidates and scores
    P->>F: combine zone, OCR, ReID evidence
    F-->>P: fused score and breakdown
    P->>C: apply thresholds and uncertainty rules
    C-->>P: ACCEPTED or UNCERTAIN
    P->>DB: store observation
    P-->>API: response JSON
    API-->>U: overlay, side panel, decision
```

## Decision Policy

The policy is not a black-box classifier. It is a thresholded decision layer built around explicit evidence.

Configuration lives in `backend/app/config.yaml`:

```yaml
tau_zone: 0.65
tau_det: 0.40
tau_ocr: 0.70
tau_reid: 0.50
tau_gap: 0.08
tau_blur: 0.60
tau_low_light: 0.35
frame_stride: 5
zone_search_topk: 20
max_zone_candidates: 20
zone_aggregate_mode: sum
max_device_matches: 5
detector_model: models/detector.pt
ocr_preprocess: true
ocr_resize_factor: 2.0
ocr_expand_ratio: 0.12
```

```mermaid
stateDiagram-v2
    [*] --> AssessQuality
    AssessQuality --> Defer: blurry / low light / glare too severe
    AssessQuality --> CheckZone: quality acceptable
    CheckZone --> Defer: zone score below tau_zone
    CheckZone --> CheckDetection: zone evidence acceptable
    CheckDetection --> Defer: no boxes or low detector confidence
    CheckDetection --> CheckOCR: candidate ROI available
    CheckOCR --> Accepted: enrolled OCR ID and conf >= tau_ocr
    CheckOCR --> CheckReID: OCR missing or weak
    CheckReID --> Accepted: score >= tau_reid and gap >= tau_gap
    CheckReID --> Defer: weak or ambiguous retrieval
    Defer --> UserFeedback: ask tap / move closer / reduce glare
    UserFeedback --> Accepted: user confirms or tap-selects
```

The core product behavior is intentional: if evidence is incomplete, the assistant should say why and ask for help.

## Implemented Components

### Backend

| Area | Files | Notes |
| --- | --- | --- |
| App entrypoint | `backend/app/main.py` | FastAPI app wiring |
| Inference routes | `backend/app/routes/infer.py` | image, video, webcam frame |
| Pipeline | `backend/app/pipeline.py` | orchestrates quality, zone, detector, OCR, ReID, fusion, policy |
| Detector | `backend/app/detector.py` | loads trained YOLOv8 detector |
| Quality | `backend/app/quality.py` | blur, brightness, glare diagnostics |
| Zone retrieval | `backend/app/zone_localizer.py` | FAISS search over zone keyframes |
| OCR | `backend/app/ocr.py` | EasyOCR/pytesseract path, preprocessing variants, ID parsing |
| ReID | `backend/app/reid.py` | embedding retrieval over device refs |
| Fusion | `backend/app/fusion.py` | combines OCR/ReID/zone evidence |
| Policy | `backend/app/policy.py` | accepts or defers |
| Tracker | `backend/app/tracker.py` | frame-to-frame continuity |
| Persistence | `backend/app/db.py` | SQLite schema and helpers |
| FAISS | `backend/app/faiss_store.py` | local vector index storage |
| Ask/evidence | `backend/app/evidence.py`, `backend/app/routes/ask.py` | rule-based answers from structured evidence |
| Debug | `backend/app/routes/debug.py` | DB and FAISS status |

### Frontend

| Area | Files | Notes |
| --- | --- | --- |
| Live demo | `frontend/src/pages/Live.jsx` | webcam, image, video workflows |
| Overlay | `frontend/src/components/OverlayCanvas.jsx` | detection boxes and interaction |
| Evidence panel | `frontend/src/components/SidePanel.jsx` | decision, OCR, ReID, zone, reasons, ask box |
| API client | `frontend/src/api.js` | backend calls |
| Device/zone pages | `frontend/src/pages/Devices.jsx`, `frontend/src/pages/Zones.jsx` | management views |

### Scripts and Experiments

| Purpose | Scripts |
| --- | --- |
| Combine detector data | `scripts/prepare_combined_detection_dataset.py` |
| Train detector | `scripts/train_baseline_detector.py` |
| Evaluate detector | `scripts/evaluate_detector.py` |
| Check backend detector | `scripts/check_backend_detector_integration.py` |
| Robustness setup | `scripts/setup_robustness_datasets.py` |
| Synthetic corruptions | `scripts/generate_synthetic_corruptions.py` |
| Classical preprocessing | `scripts/preprocess_images.py` |
| Robustness evaluation | `scripts/evaluate_preprocessing_detector.py` |
| Preprocessing previews | `scripts/preview_preprocessing_examples.py` |
| Proxy identity benchmark | `scripts/build_proxy_device_benchmark.py` |
| Proxy preview | `scripts/preview_proxy_device_benchmark.py` |

## Data Strategy

No single public dataset provides everything ValveLens needs: industrial zones, valves/gauges, exact repeated device identities, readable tags, and adverse visual conditions. The project therefore uses a deliberate multi-dataset strategy.

```mermaid
flowchart TD
    A[OpenLORIS / NYC Indoor VPR] --> Z[Zone retrieval benchmark]
    B[Valve and gauge YOLO datasets] --> D[Detector training and evaluation]
    B --> P[Proxy identity crop benchmark]
    C[ExDARK low-light dataset] --> R[Qualitative robustness examples]
    D --> M[models/detector.pt]
    P --> I[Device refs, queries, manifests]
    I --> V[OCR/ReID/fusion validation]
    R --> E[Preprocessing discussion]
```

### Zone Data

Zone localization uses public indoor-place data as a proxy for facility zones:

- OpenLORIS corridor, office, and station folders
- NYC-Indoor-VPR when available

The point is not to claim oil-facility place recognition. The point is to test the mechanism: keyframe embeddings are stored, FAISS retrieves likely zones, and zone candidates become context for identity.

### Detection Data

The detector dataset is a combined two-class YOLO dataset:

- class `0`: valve
- class `1`: gauge

The trained detector is stored locally as:

```text
models/detector.pt
```

`models/` is ignored by git, so another machine must restore this file locally.

### Identity Data

Detection data and identity data are not the same.

Detection answers:

- is there a valve or gauge?
- where is the object?

Identity answers:

- is this exact registered device `V-1023`, `V-2040`, or `PG-45`?
- does a reference/query split retrieve the same device?
- does OCR read the ID tag?
- does the policy accept or defer?

The current proxy identity benchmark is generated under:

```text
data/device_benchmark/
  devices_manifest.csv
  queries_manifest.csv
  refs/
  queries/
```

Generated data is ignored by git. Only scripts, docs, and small examples should be committed.

## Current Validation Results

### Detector and Robustness

Latest local robustness artifact: `artifacts/robustness/robustness_summary.json`.

| Condition | mAP50 | Notes |
| --- | ---: | --- |
| Original test set | `0.870` | clean baseline |
| Synthetic low light | `0.857` | small drop |
| Low light + CLAHE | `0.861` | small partial recovery |
| Low light + gamma | `0.808` | worse in this run |
| Synthetic blur | `0.823` | moderate drop |
| Blur + sharpen/CLAHE | `0.830` | slight partial recovery |
| Synthetic glare | `0.854` | small drop |
| Glare + CLAHE | `0.853` | roughly neutral |
| Synthetic low contrast | `0.830` | moderate drop |
| Low contrast + CLAHE | `0.845` | partial recovery |
| Synthetic noise | `0.601` | largest observed degradation |
| Noise + denoise/CLAHE | `0.706` | meaningful partial recovery, still below clean |

Interpretation:

- preprocessing is not a replacement for better data or training
- noise hurt the detector most in this run
- denoise+CLAHE recovered part of the lost performance for noise
- preprocessing remains an experiment and is not integrated into runtime inference

### Proxy Identity Benchmark

Latest local identity artifact: `artifacts/identity_benchmark/identity_benchmark_summary.json`.

| Metric | Value |
| --- | ---: |
| devices | `3` |
| device refs | `24` |
| device FAISS size | `24` |
| total query images | `36` |
| missing files | `0` |
| expected devices missing | `0` |
| ReID top-1 accuracy | `0.944` |
| ReID top-k accuracy | `0.944` |
| visible tag images | `26` |
| OCR attempted images | `0` |
| OCR status | `unavailable` |
| API evaluated images | `36` |
| API accepted | `0` |
| API deferred | `36` |

The OCR result should not be interpreted as an OCR algorithm failure. The current environment reports that `pytesseract` imports, but `tesseract.exe` is missing from PATH and EasyOCR is unavailable. Install/configure an OCR backend before reporting OCR exact-match rate.

The API result should also be interpreted carefully. The proxy query images are tight crops. The full runtime API expects normal scene evidence, including detector and zone context. Full-frame proxy scenes or real captured device scenes are the next step for an accepted API/Live path.

## Structured Evidence and Ask Interface

ValveLens includes an intermediate evidence layer for future interactive assistant work. This is deliberately rule-based for now.

```mermaid
flowchart TD
    A[Inference response] --> B[build_evidence]
    B --> C[Compact structured evidence]
    C --> D[Rule-based ask route]
    D --> E[Human-readable answer]

    C --> F[Future VLM input]
    F -. later .-> G[Explanation from evidence]
    H[Raw image] -. not direct source of truth .-> F
```

The future VLM should not replace the perception pipeline. It should receive ValveLens evidence:

- zone candidates
- detections
- OCR text and parsed IDs
- ReID candidates
- fusion scores
- decision status
- selected detection if the user clicked/tapped
- image quality diagnostics
- uncertainty reasons

Then it may explain the result in natural language. It should not invent a device ID that is not supported by OCR, ReID, fusion, or the enrolled database.

## Repository Structure

```text
ValveLens/
  backend/
    app/
      routes/               FastAPI routes
      cli/                  enrollment, indexing, validation, metrics
      tests/                backend tests
      pipeline.py           inference orchestration
      detector.py           YOLO detector wrapper
      zone_localizer.py     zone retrieval
      ocr.py                OCR and device ID parsing
      reid.py               device retrieval
      fusion.py             identity score fusion
      policy.py             uncertainty decision policy
      evidence.py           compact evidence for ask route
      db.py                 SQLite persistence
      faiss_store.py        vector index helpers
  frontend/
    src/
      pages/
      components/
      api.js
  scripts/
    training, dataset, robustness, and proxy benchmark scripts
  docs/
    PROJECT_STATUS.md
    NEXT_STAGE_V03.md
    DEVICE_IDENTITY_BENCHMARK.md
    ROBUSTNESS_PREPROCESSING.md
    INTERACTIVE_ASSISTANT_PLAN.md
    AGENT_CONTEXT_PACK.md
  data/
    generated datasets and benchmarks, ignored by git
  data_sources/
    downloaded and extracted source datasets, ignored by git
  models/
    local trained weights, ignored by git
  artifacts/
    generated experiment summaries and previews, ignored by git
```

## Quickstart on Windows

These commands assume the repository is at:

```powershell
D:\python_works\ValveLens
```

### Backend

```powershell
cd D:\python_works\ValveLens
python -m venv .venv
.\.venv\Scripts\activate
pip install -r backend\requirements.txt
uvicorn app.main:app --reload --port 8000 --app-dir backend
```

Check status:

```powershell
Invoke-RestMethod http://localhost:8000/debug/status
```

### Frontend

```powershell
cd D:\python_works\ValveLens\frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

### Tests

```powershell
cd D:\python_works\ValveLens\backend
pytest app\tests
```

## Core Runbooks

### 1. Import Zone Data

```powershell
cd D:\python_works\ValveLens\backend
python -m app.cli.init_db
python -m app.cli.import_openloris_zones --root "D:\python_works\ValveLens\data_sources\extracted" --max_per_zone 300 --rebuild
python -m app.cli.smoke_zones_aggregate --image "D:\python_works\ValveLens\data_sources\extracted\corridor\000\000.png" --topk 5
```

Expected direction:

- non-zero `zones_count`
- non-zero `zone_keyframes_count`
- non-zero `zone_faiss_size`

### 2. Train or Verify Detector

Prepare combined data:

```powershell
cd D:\python_works\ValveLens
python scripts\prepare_combined_detection_dataset.py
python scripts\inspect_combined_detection_dataset.py
```

Train:

```powershell
python scripts\train_baseline_detector.py --model yolov8n.pt --epochs 30 --imgsz 640 --device 0 --name valvelens_v1_cuda --copy-best
```

Evaluate:

```powershell
python scripts\evaluate_detector.py --weights models\detector.pt --data artifacts\detection_training\combined_ultralytics.yaml --split test
python scripts\check_backend_detector_integration.py
```

### 3. Build Proxy Identity Benchmark

Use this when real repeated physical device images are not available yet.

```powershell
cd D:\python_works\ValveLens
python .\scripts\build_proxy_device_benchmark.py --devices 3 --refs-per-device 8 --queries-per-device 12 --zone-id <PASTE_REAL_ZONE_ID> --seed 42 --overwrite --easy-tags
python .\scripts\preview_proxy_device_benchmark.py
```

Enroll and validate:

```powershell
cd D:\python_works\ValveLens\backend
python -m app.cli.enroll_devices_from_manifest --manifest ..\data\device_benchmark\devices_manifest.csv --refs-root ..\data\device_benchmark\refs --force-add-refs
python -m app.cli.rebuild_device_index
python -m app.cli.smoke_reid --image "..\data\device_benchmark\queries\V-1023\clean\q001.jpg" --topk 5
python -m app.cli.validate_identity_benchmark --queries-manifest ..\data\device_benchmark\queries_manifest.csv --topk 5 --out ..\artifacts\identity_benchmark
```

Run optional API-backed validation if the backend is running:

```powershell
python -m app.cli.validate_identity_benchmark --queries-manifest ..\data\device_benchmark\queries_manifest.csv --topk 5 --backend-url http://localhost:8000 --out ..\artifacts\identity_benchmark
```

### 4. Diagnose OCR

```powershell
cd D:\python_works\ValveLens\backend
python -m app.cli.check_ocr_backend
python -m app.cli.smoke_ocr --image "..\data\device_benchmark\queries\V-1023\clean\q001.jpg" --expected V-1023
```

If Tesseract is missing on Windows, install Tesseract-OCR and add this folder to PATH:

```text
C:\Program Files\Tesseract-OCR
```

Then open a new terminal and rerun the OCR check.

### 5. Run Robustness Preprocessing Experiments

```powershell
cd D:\python_works\ValveLens
python .\scripts\setup_robustness_datasets.py
python .\scripts\generate_synthetic_corruptions.py --limit 100
python .\scripts\preprocess_images.py --source data\robustness\synthetic\low_light --variant clahe --out data\robustness\preprocessed\low_light_clahe
python .\scripts\preprocess_images.py --source data\robustness\synthetic\noise --variant denoise_clahe --out data\robustness\preprocessed\noise_denoise_clahe
python .\scripts\preview_preprocessing_examples.py
python .\scripts\evaluate_preprocessing_detector.py
```

Outputs:

```text
artifacts/robustness/robustness_summary.json
artifacts/robustness/robustness_summary.csv
artifacts/robustness/preprocessing_preview/
artifacts/robustness/restoration_preview/
```

## API Overview

```text
POST /infer/image
POST /infer/video
POST /infer/webcam/frame
POST /ask
POST /feedback
POST /zones/create
POST /zones/{zone_id}/keyframes
POST /zones/rebuild_index
POST /devices/create
POST /devices/{device_id}/refs
POST /devices/rebuild_index
GET  /debug/status
```

Every inference response is logged as an observation. This gives the project traceability for later metrics, feedback, and experiment analysis.

## Example Ask Route

Run inference first, then ask from the latest or selected observation:

```powershell
$body = @{
  question = "Why are you uncertain?"
  obs_id = "<OBSERVATION_REQUEST_ID>"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -ContentType "application/json" -Body $body
```

The answer is rule-based and grounded in stored ValveLens evidence.

## Metrics and Experiment Logging

Export observations:

```powershell
cd D:\python_works\ValveLens\backend
python -m app.cli.export_metrics --out data\metrics_v03.csv --gt data\gt_sessions.json
python -m app.cli.summarize_metrics --in data\metrics_v03.csv
```

Identity benchmark outputs:

```text
artifacts/identity_benchmark/identity_benchmark_summary.json
artifacts/identity_benchmark/identity_benchmark_summary.csv
```

Robustness outputs:

```text
artifacts/robustness/robustness_summary.json
artifacts/robustness/robustness_summary.csv
```

## Development History

Recent commit history shows the evolution of the system:

```text
Add proxy device identity benchmark generation
Add identity benchmark and evidence-based ask workflow
Add robustness preprocessing experiments
Improve identity robustness and dataset docs
Add project status docs and v0.3 identity runbook
Improve identity feedback flow and surface OCR/ReID evidence
pipeline: integrate trained detector weights and verify backend loading
Prepare combined detector dataset and train CUDA YOLO baseline
data: unify valve/gauge detection dataset
Implement zone/reid pipeline upgrades with tracking and metrics export
```

This history matters because ValveLens has moved from a broad prototype to a more disciplined research system:

1. build a working modular backend/frontend
2. add zone retrieval and FAISS storage
3. train and integrate a detector
4. wire OCR/ReID/fusion/policy
5. surface evidence in the UI
6. add feedback and metrics
7. add robustness experiments
8. add device identity benchmark tooling
9. validate proxy ReID and expose current OCR/API blockers honestly

## Current v0.3 Blockers

The next milestone is an identity-aware assistant that can identify enrolled devices using OCR and/or ReID, preserve identity across nearby frames, and export useful experiment metrics.

Remaining blockers:

1. OCR backend setup on Windows
   - `pytesseract` package is installed, but `tesseract.exe` is missing unless configured locally.
   - Run `python -m app.cli.check_ocr_backend`.

2. API accepted identity path
   - proxy ReID works on tight crops
   - current API validation on those crops defers all images
   - next step is full-frame proxy scenes or real captured scenes with normal detector and zone evidence

3. Real identity validation
   - proxy benchmark is useful but synthetic
   - final validation needs repeated photos of physical devices with separate refs and queries

## Recommended Next Task

The most practical next implementation task is:

> Add a full-frame proxy scene generator that places proxy device crops into scene-like backgrounds, preserves ground-truth device IDs, and allows `/infer/image` to produce normal detector, OCR, ReID, fusion, and policy evidence.

Why this is the right next step:

- ReID already works on proxy crops
- OCR can be tested after Tesseract is installed
- the API currently defers tight crops because they are not natural runtime inputs
- full-frame proxy scenes bridge the gap between crop benchmark and live assistant behavior
- it keeps the architecture unchanged

## Documentation Map

Read these when continuing work:

| Document | Purpose |
| --- | --- |
| `docs/AGENT_CONTEXT_PACK.md` | compact handoff for new agents |
| `docs/PROJECT_STATUS.md` | current status and milestone timeline |
| `docs/NEXT_STAGE_V03.md` | v0.3 runbook |
| `docs/DEVICE_IDENTITY_BENCHMARK.md` | identity benchmark design and commands |
| `docs/ROBUSTNESS_PREPROCESSING.md` | restoration/preprocessing experiments |
| `docs/INTERACTIVE_ASSISTANT_PLAN.md` | evidence-aware ask/VLM direction |
| `data_sources/README_DATASETS.md` | local dataset inventory and import notes |

## Storage Rules

Do not commit generated or large assets:

- `data/`
- `data_sources/downloads/`
- `data_sources/extracted/`
- `backend/data/`
- `models/`
- `runs/`
- `artifacts/`
- `.venv/`
- `frontend/node_modules/`

Commit:

- source code
- scripts
- documentation
- small example manifests when force-added intentionally
- compact reproducibility notes

## License and Data Note

ValveLens combines project code with local datasets and model artifacts. Dataset archives, extracted datasets, trained weights, and generated benchmark outputs are intentionally ignored by git. Check the original dataset licenses before redistributing any raw images or trained artifacts.

## Short Summary

ValveLens is a research prototype for uncertainty-aware industrial device localization and identity verification. It is not just a detector. It is a pipeline that localizes context, detects device candidates, reads tags, retrieves enrolled references, fuses evidence, and decides whether to accept or ask. The current system has strong structural coverage and proxy ReID validation. The next v0.3 work is to make OCR operational locally and validate at least one accepted identity path on full-frame proxy or real device scenes.
