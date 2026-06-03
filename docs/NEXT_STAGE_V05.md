# Next Stage: v0.5 Interactive Assistant

Last updated: 2026-06-02

ValveLens v0.5 builds on the closed v0.3 controlled proxy identity benchmark. v0.5.1 packages the final thesis/demo evidence for an interactive assistant that answers user questions through ValveLens evidence rather than blind image guessing.

## Current implementation

- `POST /ask` is available through `backend/app/routes/ask.py`.
- Structured evidence is built by `backend/app/evidence.py`.
- Rule-based answers are implemented for common operator questions.
- VLM visual answering and VLM-only structured demo inference exist in `backend/app/vlm_assistant.py`.
- VLM execution is controlled by `backend/app/config.yaml` and external environment credentials.
- The Live frontend side panel includes a question box, Describe Image action, answer panel, grouped ReID evidence, and decision reason bullets.
- Local demo sample picker exists through `/demo/samples` and `/demo/infer_sample`.
- Demo Flow mode exists at `/demo-flow` for screenshot capture.
- `frontend/src/components/ZoneMap.jsx` renders zone context as real layout coordinates when available or as a schematic fallback.
- `GET /demo/zone-layout` loads `data/demo_zone_layout.json` or `data/demo_zone_layout.example.json`.
- Image inference audit CLI exists at `backend/app/cli/audit_inference_image.py`.
- A reproducible assistant demo CLI exists at `backend/app/cli/demo_assistant_queries.py`.
- Demo artifacts are saved under `artifacts/v05_assistant_demo`.
- Final result summaries are saved under `artifacts/final_results`.

## Evidence contract

Every assistant answer should be grounded in:

- zone candidates
- detections and bounding boxes
- class names and detector confidence
- OCR text and parsed device IDs
- ReID top-k candidates
- fused identity result
- decision status and reason
- image-quality diagnostics
- selected detection when available

Detector display contract:

- every detection should expose `class_id`
- every detection should expose raw model `class_name`
- overlay labels should use `class_name confidence%`
- `unknown` is reserved for true missing class names or fallback detector output

ReID contract:

- raw reference matches are grouped by `device_id`
- policy gap checks compare unique device IDs, not repeated references
- if all strong matches are the same device ID, that alone is not ambiguity

## v0.5.1 final package

- Assistant CLI default questions now match the thesis/demo set:
  - `What is this?`
  - `Where is V-1023?`
  - `Which devices are visible?`
  - `Why are you uncertain?`
  - `What should I do next?`
  - `What tag did you read?`
- The assistant report, JSON, CSV, and thesis section are generated in `artifacts/v05_assistant_demo`.
- Identity, assistant, detector, and robustness summaries are collected in `artifacts/final_results`.

## Remaining next steps

1. Capture `/demo-flow` screenshots for normal model mode and VLM-only demo mode.
2. Capture Live page screenshots for operator-style interaction if needed.
3. Keep VLM prompt behavior practical: direct visual answers for normal questions, evidence details only when asked.
4. Validate the assistant on real full-frame device photos after those images are collected.
5. Add a small browser-level demo recording for the final presentation.

## Demo Flow screenshot guide

Open:

```text
http://localhost:5173/demo-flow
```

Capture these six cards:

1. Input: uploaded image or selected sample.
2. Detection: image with overlay labels.
3. Evidence: OCR, ReID, fusion, and decision.
4. Zone Context: `Zone context schematic`.
5. Ask / Query: quick question and visible answer panel.
6. Confirm: Confirm, Wrong, and Tap Select feedback controls.

The zone map is schematic unless real coordinates exist in `data/demo_zone_layout.json`. Scores greater than `1.0` are shown as raw scores, not percentages.

## Demo upload folders

Use these folders first during demos:

- `data/device_benchmark/queries/`
- `data/device_benchmark/manual_v1023/queries/`
- `data/device_benchmark/fullframe_demo/`
- `data/detection/combined/test/images/`
- `data/detection/industrial_multiclass/test/images/`

If an uploaded industrial image looks wrong, run:

```powershell
cd D:\python_works\ValveLens\backend
python -m app.cli.audit_inference_image --image "PATH_TO_IMAGE" --model models\detector.pt --also-model models\detector_multiclass.pt
python -m app.cli.check_vlm_backend
python -m app.cli.smoke_vlm_assistant --image "PATH_TO_IMAGE" --question "What do you see in this image?" --use-vlm
```

Interpretation rule:

- detector failure means YOLO did not return useful boxes
- VLM visual description can explain scene content but cannot assign exact device identity
- exact identity still requires ValveLens evidence from OCR, ReID, fusion, or user confirmation

## Do not do yet

- Do not replace `models/detector.pt`.
- Do not retrain YOLO for this milestone.
- Do not add new datasets.
- Do not integrate runtime preprocessing.
- Do not allow the VLM to override strong ValveLens evidence silently.
