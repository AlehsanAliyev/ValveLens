import { useState } from "react";

import { createDevice, uploadDeviceRefs, rebuildDeviceIndex } from "../api";
import DeviceCard from "../components/DeviceCard";

export default function Devices() {
  const [deviceId, setDeviceId] = useState("");
  const [zoneId, setZoneId] = useState("");
  const [deviceType, setDeviceType] = useState("valve");
  const [description, setDescription] = useState("");
  const [files, setFiles] = useState([]);
  const [createdDevice, setCreatedDevice] = useState(null);
  const [status, setStatus] = useState("");

  async function handleCreate() {
    try {
      const res = await createDevice({
        device_id: deviceId,
        zone_id: zoneId,
        device_type: deviceType,
        description,
      });
      setCreatedDevice({ deviceId: res.device_id, zoneId, deviceType });
      setStatus("Device created.");
    } catch (err) {
      setStatus(err.message);
    }
  }

  async function handleUpload() {
    try {
      const res = await uploadDeviceRefs(deviceId, files);
      setStatus(`Uploaded ${res.count} device refs.`);
    } catch (err) {
      setStatus(err.message);
    }
  }

  async function handleRebuild() {
    const res = await rebuildDeviceIndex();
    setStatus(`Rebuilt device index with ${res.count} items.`);
  }

  return (
    <div>
      <div className="page-title">Device Enrollment</div>
      <p className="subtitle">
        Register devices and upload reference images for re-identification.
      </p>
      <div className="grid grid-3">
        <div className="card">
          <h3>Create Device</h3>
          <div className="field">
            <label>Device ID</label>
            <input className="input" value={deviceId} onChange={(e) => setDeviceId(e.target.value)} />
          </div>
          <div className="field">
            <label>Zone ID</label>
            <input className="input" value={zoneId} onChange={(e) => setZoneId(e.target.value)} />
          </div>
          <div className="field">
            <label>Type</label>
            <select value={deviceType} onChange={(e) => setDeviceType(e.target.value)}>
              <option value="valve">valve</option>
              <option value="gauge">gauge</option>
              <option value="panel">panel</option>
              <option value="tag">tag</option>
              <option value="unknown">unknown</option>
            </select>
          </div>
          <div className="field">
            <label>Description</label>
            <input
              className="input"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
          </div>
          <button className="button" onClick={handleCreate}>
            Create Device
          </button>
        </div>
        <div className="card">
          <h3>Upload References</h3>
          <div className="field">
            <label>Device ID</label>
            <input className="input" value={deviceId} onChange={(e) => setDeviceId(e.target.value)} />
          </div>
          <div className="field">
            <label>Images</label>
            <input type="file" multiple accept="image/*" onChange={(e) => setFiles(e.target.files)} />
          </div>
          <button className="button secondary" onClick={handleUpload}>
            Upload
          </button>
        </div>
        <div className="card">
          <h3>Rebuild Index</h3>
          <p>Refresh FAISS with newly added device references.</p>
          <button className="button ghost" onClick={handleRebuild}>
            Rebuild
          </button>
        </div>
      </div>
      {createdDevice && (
        <div style={{ marginTop: 20 }}>
          <DeviceCard {...createdDevice} />
        </div>
      )}
      {status && <div className="mono" style={{ marginTop: 12 }}>{status}</div>}
    </div>
  );
}
