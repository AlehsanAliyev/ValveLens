import { useState } from "react";

const DEFAULT_THRESHOLDS = {
  tauZone: 0.65,
  tauDet: 0.4,
  tauOcr: 0.7,
  tauReid: 0.5,
  tauGap: 0.08,
};

function buildDecisionReasons(response) {
  if (!response?.decision || response.decision.status === "ACCEPTED") {
    return [];
  }
  if (response.decision.reasons?.length) {
    return response.decision.reasons;
  }

  const reasons = [];
  const zoneTop1 = response.zone?.top1;
  const quality = response.quality || {};
  const detections = response.detections || [];

  if (!zoneTop1 || zoneTop1.score < DEFAULT_THRESHOLDS.tauZone) {
    reasons.push("zone confidence below threshold");
  }
  if (quality.is_blurry) {
    reasons.push("blur score below threshold");
  }
  if (quality.is_low_light) {
    reasons.push("brightness below threshold");
  }

  const bestDet = detections.reduce(
    (best, det) => Math.max(best, Number(det.conf || 0)),
    0
  );
  if (detections.length === 0 || bestDet < DEFAULT_THRESHOLDS.tauDet) {
    reasons.push("device detection below threshold");
  }

  const ocrDetections = detections.filter((det) => det.ocr?.text);
  const strongOcr = ocrDetections.some(
    (det) => Number(det.ocr?.conf || 0) >= DEFAULT_THRESHOLDS.tauOcr
  );
  if (ocrDetections.length === 0) {
    reasons.push("no readable OCR tag");
  } else if (!strongOcr) {
    reasons.push("OCR confidence below threshold");
  } else {
    reasons.push("OCR text did not match an enrolled device");
  }

  const topMatchSets = detections
    .map((det) => det.reid?.top_matches || [])
    .filter((matches) => matches.length > 0);
  if (topMatchSets.length === 0) {
    reasons.push("no ReID matches from device index");
  } else {
    const bestMatch = Math.max(
      ...topMatchSets.map((matches) => Number(matches[0]?.score || 0))
    );
    const ambiguous = topMatchSets.some((matches) => {
      if (matches.length < 2) return false;
      return (
        Number(matches[0]?.score || 0) - Number(matches[1]?.score || 0) <
        DEFAULT_THRESHOLDS.tauGap &&
        matches[0]?.device_id !== matches[1]?.device_id
      );
    });
    if (bestMatch < DEFAULT_THRESHOLDS.tauReid) {
      reasons.push("ReID score below threshold");
    }
    if (ambiguous) {
      reasons.push("ReID identity margin low");
    }
  }

  return [...new Set(reasons)];
}

export default function SidePanel({
  response,
  mode,
  onModeChange,
  onConfirm,
  onReject,
  onAsk,
  askResult,
  askStatus,
  selectedDetectionId,
}) {
  const [question, setQuestion] = useState("");
  const zoneCandidates = response?.zone?.candidates || [];
  const zoneTop1 = response?.zone?.top1;
  const decision = response?.decision;
  const detections = response?.detections || [];
  const decisionReasons = buildDecisionReasons(response);
  const isVlmOnly = detections.some(
    (det) => det.fused?.score_breakdown?.mode === "vlm_only_demo"
  );
  async function handleAskSubmit(event) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || !onAsk) return;
    await onAsk(trimmed, false);
  }

  async function handleDescribeImage() {
    if (!response || !onAsk) return;
    setQuestion("What do you see in this image?");
    await onAsk("What do you see in this image?", true);
  }

  return (
    <div className="panel">
      <div className="list">
        <div>
          <div className="pill">Zone Top1</div>
          <div style={{ marginTop: 8 }}>
            {zoneTop1 ? (
              <div className="row" style={{ alignItems: "center" }}>
                <span className="mono">{zoneTop1.zone_name || zoneTop1.zone_id}</span>
                <span className="pill">{(zoneTop1.score * 100).toFixed(0)}%</span>
              </div>
            ) : (
              <div className="mono">No top zone yet</div>
            )}
          </div>
        </div>
        <div>
          <div className="pill">Zone Candidates</div>
          <div className="list" style={{ marginTop: 8 }}>
            {zoneCandidates.length === 0 && <div className="mono">No zones yet</div>}
            {zoneCandidates.map((z) => (
              <div key={z.zone_id} className="row" style={{ alignItems: "center" }}>
                <span className="mono">{z.zone_name || z.zone_id}</span>
                <span className="pill">{(z.score * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="pill">Decision</div>
          {isVlmOnly && (
            <div style={{ marginTop: 8 }}>
              <span className="pill">VLM-only demo estimate</span>
            </div>
          )}
          {decision?.status && (
            <div className="mono" style={{ marginTop: 6 }}>
              Decision: {decision.status}
            </div>
          )}
          <div style={{ marginTop: 6 }}>
            {decision ? decision.message : "Awaiting inference..."}
          </div>
          {decision?.selected_device && (
            <div className="mono" style={{ marginTop: 8 }}>
              selected: {decision.selected_device.device_id} ({(decision.selected_device.score * 100).toFixed(0)}%)
            </div>
          )}
          {decisionReasons.length > 0 && (
            <ul className="mono" style={{ marginTop: 8, paddingLeft: 18 }}>
              {decisionReasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          )}
          {decision?.next_action && (
            <div className="mono" style={{ marginTop: 8 }}>
              next: {decision.next_action}
            </div>
          )}
          {response && decision && decision.status !== "ACCEPTED" && (
            <div className="mono" style={{ marginTop: 8 }}>
              Detector is uncertain. Use VLM description or select object manually.
            </div>
          )}
        </div>
        <div>
          <div className="pill">Identity Evidence</div>
          <div className="list" style={{ marginTop: 8 }}>
            {detections.length === 0 && (
              <div className="mono">No detections yet</div>
            )}
            {detections.map((det) => {
              const hasOcr = det.ocr?.text;
              const conf = typeof det.ocr?.conf === "number"
                ? ` (${(det.ocr.conf * 100).toFixed(0)}%)`
                : "";
              const topMatches = det.reid?.top_matches?.slice(0, 3) || [];
              const fusedDevice = det.fused?.device_id;
              const fusedScore = typeof det.fused?.final_score === "number"
                ? ` (${(det.fused.final_score * 100).toFixed(0)}%)`
                : "";
              return (
                <div key={det.det_id} className="mono">
                  <div>{det.class_name || det.cls || "unknown"} {det.track_id ? `track ${det.track_id}` : ""}</div>
                  <div>ocr: {hasOcr ? `${det.ocr.text}${conf}` : "none"}</div>
                  <div>
                    reid: {topMatches.length
                      ? topMatches
                          .map((m) => {
                            const count = m.ref_count ? ` refs:${m.ref_count}` : "";
                            return `${m.device_id} ${(m.score * 100).toFixed(0)}%${count}`;
                          })
                          .join(" | ")
                      : "none"}
                  </div>
                  <div>fused: {fusedDevice ? `${fusedDevice}${fusedScore}` : "none"}</div>
                </div>
              );
            })}
          </div>
        </div>
        <div>
          <div className="pill">Assistant / Visual Understanding</div>
          {selectedDetectionId && (
            <div className="mono" style={{ marginTop: 8 }}>
              selected detection: {selectedDetectionId}
            </div>
          )}
          <form onSubmit={handleAskSubmit} style={{ marginTop: 8 }}>
            <div className="field">
              <input
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="What is this?"
                disabled={!response}
              />
            </div>
            <div className="controls" style={{ marginTop: 8 }}>
              <button className="button" disabled={!response || !question.trim()}>
                Ask
              </button>
              <button
                type="button"
                className="button secondary"
                disabled={!response}
                onClick={handleDescribeImage}
              >
                Describe image
              </button>
            </div>
          </form>
          {askStatus && (
            <div className="mono" style={{ marginTop: 8 }}>{askStatus}</div>
          )}
          {askResult && (
            <div className="list" style={{ marginTop: 8 }}>
              <div>{askResult.answer}</div>
              <div className="row" style={{ alignItems: "center" }}>
                <span className="pill">{askResult.mode || "rule_based"}</span>
                <span className="mono">
                  confidence: {(Number(askResult.confidence || 0) * 100).toFixed(0)}%
                </span>
              </div>
              {askResult.evidence_used?.length > 0 && (
                <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
                  {askResult.evidence_used.map((item) => (
                    <span key={item} className="pill">{item}</span>
                  ))}
                </div>
              )}
              <div>
                next: {askResult.recommended_next_action || "none"}
              </div>
              {askResult.uncertainty_reason && (
                <div className="mono">uncertain: {askResult.uncertainty_reason}</div>
              )}
              {askResult.fallback_reason && (
                <div className="mono">fallback: {askResult.fallback_reason}</div>
              )}
            </div>
          )}
        </div>
        <div>
          <div className="pill">Feedback</div>
          <div className="controls" style={{ marginTop: 8 }}>
            <button className="button" onClick={onConfirm} disabled={!response}>
              Confirm
            </button>
            <button className="button ghost" onClick={onReject} disabled={!response}>
              Wrong
            </button>
            <button
              className={`button ${mode === "tap" ? "secondary" : "ghost"}`}
              onClick={() => onModeChange(mode === "tap" ? "view" : "tap")}
              disabled={!response}
            >
              Tap Select
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
