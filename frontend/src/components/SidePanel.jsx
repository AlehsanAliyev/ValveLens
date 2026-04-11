export default function SidePanel({
  response,
  mode,
  onModeChange,
  onConfirm,
  onReject,
}) {
  const zoneCandidates = response?.zone?.candidates || [];
  const zoneTop1 = response?.zone?.top1;
  const decision = response?.decision;
  const detections = response?.detections || [];
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
          <div style={{ marginTop: 6 }}>
            {decision ? decision.message : "Awaiting inference..."}
          </div>
          {decision?.selected_device && (
            <div className="mono" style={{ marginTop: 8 }}>
              selected: {decision.selected_device.device_id} ({(decision.selected_device.score * 100).toFixed(0)}%)
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
                  <div>{det.cls} {det.track_id ? `track ${det.track_id}` : ""}</div>
                  <div>ocr: {hasOcr ? `${det.ocr.text}${conf}` : "none"}</div>
                  <div>
                    reid: {topMatches.length
                      ? topMatches
                          .map((m) => `${m.device_id} ${(m.score * 100).toFixed(0)}%`)
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
          <div className="pill">Feedback</div>
          <div className="controls" style={{ marginTop: 8 }}>
            <button className="button" onClick={onConfirm}>
              Confirm
            </button>
            <button className="button ghost" onClick={onReject}>
              Wrong
            </button>
            <button
              className={`button ${mode === "tap" ? "secondary" : "ghost"}`}
              onClick={() => onModeChange(mode === "tap" ? "view" : "tap")}
            >
              Tap Select
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
