# ValveLens Interactive Assistant

Last updated: 2026-05-23

This document describes the ValveLens v0.5.1 interactive assistant layer. The assistant is evidence-first: it answers user questions from structured ValveLens perception output instead of blindly guessing from an image.

## Design rule

The assistant must use ValveLens evidence before language generation:

- zone candidates
- detections and bounding boxes
- detected class names and detector confidence
- OCR text and parsed device IDs
- ReID top-k candidates
- fused identity result
- final decision status
- accepted/deferred reason
- uncertainty/blocking reasons
- image-quality diagnostics
- selected detection when the user taps an object

The VLM layer is not allowed to invent device IDs, locations, or confidence. If evidence is weak, it must say uncertain and recommend the next action.

## Backend components

- `backend/app/evidence.py` builds a compact evidence object from an inference observation.
- `backend/app/routes/ask.py` exposes `POST /ask`.
- `backend/app/vlm_assistant.py` provides the VLM scaffold and rule fallback.
- `backend/app/config.yaml` controls assistant settings.

The current local demo configuration enables DeepInfra visual understanding when `.env` contains a DeepInfra endpoint and key:

```yaml
assistant:
  enable_vlm: true
  provider: deepinfra
  model: Qwen/Qwen2.5-VL-32B-Instruct
  include_image: true
  max_tokens: 300
  use_rule_fallback: true
```

The provider still falls back to rule-based answers if the model call fails, credentials are absent, or the image path is unavailable.

## API

Endpoint:

```text
POST /ask
```

Example request:

```json
{
  "question": "What is this?",
  "session_id": "session-001",
  "observation_id": "obs-001",
  "selected_detection_id": "det-1",
  "use_vlm": false
}
```

Example response:

```json
{
  "answer": "This is likely V-1023. ValveLens accepted this identity from the available OCR/ReID evidence.",
  "confidence": 0.82,
  "mode": "rule_based",
  "evidence_used": ["decision", "ocr", "reid", "fusion"],
  "recommended_next_action": "Identified device V-1023 via OCR.",
  "uncertainty_reason": ""
}
```

## Supported questions

The rule-based assistant currently handles:

- `What is this?`
- `Where is V-1023?`
- `Which devices are visible?`
- `Why are you uncertain?`
- `What should I do next?`
- `What tag did you read?`
- `What do you see in this image?` by returning detector/evidence summaries, or a clear VLM fallback reason when visual inspection is unavailable.

The demo CLI can also include `What are the top candidates?` with `--include-candidates`.

## Demo validation

The v0.5.1 assistant demo was validated against stored inference observations from the controlled proxy benchmark. The demo includes both:

- an accepted identity observation, where `PG-45` was accepted through OCR-backed ValveLens evidence
- an uncertain observation, where no detector boxes, OCR tag, or ReID candidates were available

Artifacts:

- `artifacts/v05_assistant_demo/assistant_demo_report.md`
- `artifacts/v05_assistant_demo/assistant_demo_report.json`
- `artifacts/v05_assistant_demo/example_questions.csv`
- `artifacts/v05_assistant_demo/thesis_assistant_section.md`

The final CLI run used the local route call in default rule-based mode. VLM provider execution can be smoke-tested with `--use-vlm`; if the configured provider is unavailable, the route falls back to `rule_based` answers and reports `vlm_status`.

Demo command:

```powershell
cd D:\python_works\ValveLens\backend
python -m app.cli.demo_assistant_queries --observation-ids 3fa6485b-b5a1-43c0-b0cf-9a167495bb26 80a89ea6-6e33-4ea1-9824-661171ce8b72 --out ..\artifacts\v05_assistant_demo
```

## Frontend behavior

The Live page side panel includes:

- a question input
- an Ask button
- answer text
- mode indicator: `rule_based` or `vlm`
- evidence chips
- recommended next action
- uncertainty reason
- a `Describe image` quick button that requests visual-understanding mode
- grouped ReID candidates by unique `device_id`, with reference counts
- decision reasons as bullets instead of a single vague message

The image workflow also includes a demo sample picker backed by `/demo/samples`. Use it to avoid uploading from arbitrary folders during thesis demos.

Recommended demo folders:

- `data/device_benchmark/queries/`
- `data/device_benchmark/manual_v1023/queries/`
- `data/device_benchmark/fullframe_demo/`
- `data/detection/combined/test/images/`
- `data/detection/industrial_multiclass/test/images/`

For object-specific questions, tap-select a detection first, then ask `What is this?`.

## VLM mode

`backend/app/vlm_assistant.py` is intentionally gated by configuration and environment variables. It loads local `.env` values for demo runs but never hardcodes or prints credentials.

Current behavior:

- if `assistant.enable_vlm` is `false`, the system uses the rule-based fallback
- if credentials or model are missing, the system uses the rule-based fallback
- provider execution is implemented through an OpenAI-compatible client
- the local DeepInfra default model is `Qwen/Qwen2.5-VL-32B-Instruct`
- `DEEPINFRA_ENDPOINT` may be `https://api.deepinfra.com/v1` or `https://api.deepinfra.com/v1/openai`; the runtime normalizes the first form
- demo validation confirms the fallback path returns usable answers when VLM is requested but unavailable
- environment variables are used only; key values are never printed

Supported environment variables:

- `OPENAI_API_KEY`
- `DEEPINFRA_TOKEN`
- `DEEPINFRA_API_KEY` as a compatibility alias
- `DEEPINFRA_ENDPOINT`
- `VALVELENS_ENABLE_VLM`
- `VALVELENS_VLM_PROVIDER`
- `VALVELENS_VLM_MODEL` if the model is not set in config

The VLM may describe visible scene content, for example `industrial pipe and valve assembly`, `flanged pipe connection`, or `no readable tag visible`. It must not assign exact device identity unless OCR/ReID/fusion evidence supports that ID.

## Inference Audit

Use the audit CLI when a screenshot looks wrong in the UI:

```powershell
cd D:\python_works\ValveLens\backend
python -m app.cli.audit_inference_image --image "PATH_TO_IMAGE" --model models\detector.pt --also-model models\detector_multiclass.pt
```

It writes:

- `artifacts/final_audit/image_inference_audit.md`
- `artifacts/final_audit/image_inference_audit.json`

This separates detector failure, class-name mapping failure, quality-policy failure, OCR failure, ReID ambiguity, and UI display problems.

Required prompt rule for future VLM provider execution:

```text
You are an assistant for ValveLens. Answer using only the provided ValveLens evidence.
Do not invent device IDs, locations, or confidence. If evidence is weak, say uncertain
and recommend the next action.
```

## Limitations

- The assistant depends on a stored inference observation. Run `/infer/image`, `/infer/video`, or `/infer/webcam/frame` first.
- Relative location is estimated from detection boxes in the current evidence, not from a calibrated scene map.
- VLM provider execution requires a reachable configured provider and a stored/uploaded image path.
- The assistant does not replace YOLO detection, OCR, ReID, fusion, or the uncertainty policy.
- Real facility validation still requires real repeated device photos and full-frame industrial scenes.

## Commands

Backend tests:

```powershell
cd D:\python_works\ValveLens\backend
pytest app\tests
```

Frontend build:

```powershell
cd D:\python_works\ValveLens\frontend
npm run build
```
