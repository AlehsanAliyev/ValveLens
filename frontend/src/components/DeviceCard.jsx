export default function DeviceCard({ deviceId, zoneId, deviceType }) {
  return (
    <div className="card">
      <div className="mono">{deviceId}</div>
      <div style={{ marginTop: 6 }}>{deviceType}</div>
      <div style={{ color: "#6f6256" }}>Zone: {zoneId}</div>
    </div>
  );
}
