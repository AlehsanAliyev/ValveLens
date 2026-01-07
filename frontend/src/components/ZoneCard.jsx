export default function ZoneCard({ zoneId, name, description }) {
  return (
    <div className="card">
      <div className="mono">{zoneId}</div>
      <div style={{ fontWeight: 600, marginTop: 6 }}>{name || "Zone"}</div>
      <div style={{ color: "#6f6256", marginTop: 4 }}>{description || ""}</div>
    </div>
  );
}
