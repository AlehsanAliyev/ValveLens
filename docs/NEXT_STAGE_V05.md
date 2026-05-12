# Next Stage: v0.5 Interactive Assistant

Last updated: 2026-05-12

ValveLens v0.5 builds on the closed v0.3 controlled proxy identity benchmark. The goal is an interactive assistant that answers user questions through ValveLens evidence rather than blind image guessing.

## Current implementation

- `POST /ask` is available through `backend/app/routes/ask.py`.
- Structured evidence is built by `backend/app/evidence.py`.
- Rule-based answers are implemented for common operator questions.
- VLM scaffolding exists in `backend/app/vlm_assistant.py`.
- VLM execution is disabled by default in `backend/app/config.yaml`.
- The Live frontend side panel includes a question box and answer panel.

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

## Near-term tasks

1. Capture UI screenshots of the Ask panel on accepted and uncertain examples.
2. Add a small scripted API demo that sends fixed `/ask` questions to known observations.
3. Enable a real VLM provider only after credentials and model settings are configured outside git.
4. Keep the prompt evidence-bound and test that the VLM does not invent device IDs.
5. Validate the assistant on real full-frame device photos after those images are collected.

## Do not do yet

- Do not replace `models/detector.pt`.
- Do not retrain YOLO for this milestone.
- Do not add new datasets.
- Do not integrate runtime preprocessing.
- Do not allow the VLM to override strong ValveLens evidence silently.
