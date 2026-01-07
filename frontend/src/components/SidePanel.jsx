export default function SidePanel({
  response,
  mode,
  onModeChange,
  onConfirm,
  onReject,
}) {
  const zoneCandidates = response?.zone?.candidates || [];
  const decision = response?.decision;
  return (
    <div className="panel">
      <div className="list">
        <div>
          <div className="pill">Zone Candidates</div>
          <div className="list" style={{ marginTop: 8 }}>
            {zoneCandidates.length === 0 && <div className="mono">No zones yet</div>}
            {zoneCandidates.map((z) => (
              <div key={z.zone_id} className="row" style={{ alignItems: "center" }}>
                <span className="mono">{z.zone_id}</span>
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
