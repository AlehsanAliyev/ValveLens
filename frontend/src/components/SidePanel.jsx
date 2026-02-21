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
        </div>
        <div>
          <div className="pill">Detections OCR</div>
          <div className="list" style={{ marginTop: 8 }}>
            {detections.length === 0 && (
              <div className="mono">No detections yet</div>
            )}
            {detections.map((det) => {
              const hasOcr = det.ocr?.text;
              const conf = typeof det.ocr?.conf === "number"
                ? ` (${(det.ocr.conf * 100).toFixed(0)}%)`
                : "";
              return (
                <div key={det.det_id} className="mono">
                  {det.cls}: {hasOcr ? `${det.ocr.text}${conf}` : "OCR none"}
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
