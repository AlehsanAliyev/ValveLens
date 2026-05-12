# ValveLens Interactive Assistant

Last updated: 2026-05-12

This document describes the ValveLens v0.5 interactive assistant layer. The assistant is evidence-first: it answers user questions from structured ValveLens perception output instead of blindly guessing from an image.

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

The current default is rule-based mode. VLM execution is disabled by default:

```yaml
assistant:
  enable_vlm: false
  provider: env_configured
  model: null
  max_tokens: 300
  use_rule_fallback: true
```

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
- `Where is this?`
- `Where is V-1023?`
- `Which devices are visible?`
- `Why are you uncertain?`
- `What should I do next?`
- `What tag did you read?`
- `What are the top candidates?`

## Frontend behavior

The Live page side panel includes:

- a question input
- an Ask button
- answer text
- mode indicator: `rule_based` or `vlm`
- evidence chips
- recommended next action
- uncertainty reason

For object-specific questions, tap-select a detection first, then ask `What is this?`.

## VLM mode

`backend/app/vlm_assistant.py` is intentionally gated by configuration and environment variables. It never hardcodes credentials.

Current behavior:

- if `assistant.enable_vlm` is `false`, the system uses the rule-based fallback
- if credentials or model are missing, the system uses the rule-based fallback
- provider execution is scaffolded but not enabled by default

Required prompt rule for future VLM provider execution:

```text
You are an assistant for ValveLens. Answer using only the provided ValveLens evidence.
Do not invent device IDs, locations, or confidence. If evidence is weak, say uncertain
and recommend the next action.
```

## Limitations

- The assistant depends on a stored inference observation. Run `/infer/image`, `/infer/video`, or `/infer/webcam/frame` first.
- Relative location is estimated from detection boxes in the current evidence, not from a calibrated scene map.
- VLM provider execution is not enabled by default.
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
