# Next Stage: v0.5 Interactive Assistant

Last updated: 2026-05-15

ValveLens v0.5 builds on the closed v0.3 controlled proxy identity benchmark. v0.5.1 packages the final thesis/demo evidence for an interactive assistant that answers user questions through ValveLens evidence rather than blind image guessing.

## Current implementation

- `POST /ask` is available through `backend/app/routes/ask.py`.
- Structured evidence is built by `backend/app/evidence.py`.
- Rule-based answers are implemented for common operator questions.
- VLM scaffolding exists in `backend/app/vlm_assistant.py`.
- VLM execution is disabled by default in `backend/app/config.yaml`.
- The Live frontend side panel includes a question box and answer panel.
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

1. Capture UI screenshots of the Ask panel on accepted and uncertain examples.
2. Enable a real VLM provider only after credentials and model settings are configured outside git.
3. Keep the prompt evidence-bound and test that the VLM does not invent device IDs.
4. Validate the assistant on real full-frame device photos after those images are collected.
5. Add a small browser-level demo recording for the final presentation.

## Do not do yet

- Do not replace `models/detector.pt`.
- Do not retrain YOLO for this milestone.
- Do not add new datasets.
- Do not integrate runtime preprocessing.
- Do not allow the VLM to override strong ValveLens evidence silently.
