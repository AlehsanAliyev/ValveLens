# Interactive Assistant Plan

ValveLens should become an interactive industrial perception assistant, but the perception pipeline remains the source of truth. A language model may later explain or rephrase the result, but it should not replace zone localization, detection, OCR, ReID, fusion, or the uncertainty policy.

## Current Direction

The next milestone is still v0.3 identity awareness:

1. Enroll a small set of devices such as `V-1023`, `V-2040`, and `PG-45`.
2. Add 3 to 10 reference images per device.
3. Rebuild the device FAISS index.
4. Validate OCR on a readable printed tag.
5. Validate ReID top-k on enrolled references.
6. Run one image/webcam session that reaches `ACCEPTED`.
7. Export metrics after the session.

The assistant interface should grow around that evidence. It should answer questions such as:

- What is this?
- Where is V-1023?
- Why are you uncertain?
- What should I do next?
- Which devices are visible?

## Structured Evidence Layer

The evidence object is the contract between perception and interaction. It should stay compact and machine-readable.

It contains:

- input and request ID
- image-quality diagnostics
- zone top-1 and candidates
- detections with class, confidence, bounding box, and track ID
- OCR text and parsed device IDs
- ReID top matches
- fusion device ID, score, and score breakdown
- decision status and selected device
- selected detection, if the user tapped or the policy highlighted one
- uncertainty reasons

This gives a future VLM facts to explain without guessing from the image alone.

## Implemented Now

Minimal assistant support is rule-based:

- `backend/app/evidence.py` builds compact evidence from an inference response.
- `backend/app/routes/ask.py` answers questions from stored observations using rules.
- `frontend/src/components/SidePanel.jsx` includes a small question box.
- `backend/app/cli/validate_identity_demo.py` checks local identity-demo readiness.

No VLM is called. No runtime preprocessing is added. YOLO is not retrained.

## Next Implementation Sequence

### 1. Finish v0.3 Identity Demo

Use existing CLI tools:

```powershell
cd d:\python_works\ValveLens\backend
python -m app.cli.create_device --device_id "V-1023" --zone_id "<ZONE_ID>" --type valve --desc "Printed test valve"
python -m app.cli.create_device --device_id "V-2040" --zone_id "<ZONE_ID>" --type valve --desc "Second printed test valve"
python -m app.cli.create_device --device_id "PG-45" --zone_id "<ZONE_ID>" --type gauge --desc "Printed pressure gauge"
python -m app.cli.add_device_refs --device_id "V-1023" --folder "D:\data\devices\V-1023"
python -m app.cli.add_device_refs --device_id "V-2040" --folder "D:\data\devices\V-2040"
python -m app.cli.add_device_refs --device_id "PG-45" --folder "D:\data\devices\PG-45"
python -m app.cli.rebuild_device_index
```

Validate readiness:

```powershell
python -m app.cli.validate_identity_demo
python -m app.cli.validate_identity_demo --ocr-image "D:\data\samples\V-1023_tag.jpg" --ocr-expected-id "V-1023"
python -m app.cli.validate_identity_demo --reid-image "D:\data\devices\V-1023\sample.jpg" --reid-expected-id "V-1023"
python -m app.cli.validate_identity_demo --infer-image "D:\data\samples\V-1023_live.jpg" --infer-expected-id "V-1023"
```

### 2. Use Ask Route After Inference

Start backend:

```powershell
cd d:\python_works\ValveLens
uvicorn app.main:app --reload --port 8000 --app-dir backend
```

Ask from the latest or selected observation:

```powershell
$body = @{
  question = "Why are you uncertain?"
  obs_id = "<OBSERVATION_REQUEST_ID>"
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -ContentType "application/json" -Body $body
```

### 3. Export Metrics

```powershell
cd d:\python_works\ValveLens\backend
python -m app.cli.export_metrics --out data\metrics_v03.csv --gt data\gt_sessions.json
python -m app.cli.summarize_metrics --in data\metrics_v03.csv
```

## Later VLM Use

A VLM should be added only after v0.3 identity works with real enrolled devices. The VLM should receive:

- the structured evidence object
- the user question
- optionally a cropped selected ROI, if needed

The VLM should produce:

- a clearer natural-language answer
- a short explanation of evidence
- a recommended user action

The VLM should not:

- override an `UNCERTAIN` policy decision
- invent a device ID not present in OCR, ReID, fusion, or the enrolled database
- claim clean-image performance under degraded conditions
- bypass feedback or metrics logging

## Risks

- The identity demo still depends on captured reference images.
- OCR acceptance needs large, high-contrast printed tags.
- ReID quality depends on crop similarity and the embedding model.
- The current uncertainty thresholds are heuristic.
- The frontend can ask questions, but answers are only as good as the latest stored inference evidence.

## Recommended Next Task

Enroll the first three demo devices, rebuild the device index, and run:

```powershell
python -m app.cli.validate_identity_demo --infer-image "D:\data\samples\V-1023_live.jpg" --infer-expected-id "V-1023"
```

That is the shortest path to proving v0.3 with a real accepted identity decision.
