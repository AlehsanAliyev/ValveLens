import { useState } from "react";

import { createZone, uploadZoneKeyframes, rebuildZoneIndex } from "../api";
import ZoneCard from "../components/ZoneCard";

export default function Zones() {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [zoneId, setZoneId] = useState("");
  const [files, setFiles] = useState([]);
  const [createdZone, setCreatedZone] = useState(null);
  const [status, setStatus] = useState("");

  async function handleCreate() {
    try {
      const res = await createZone({ name, description });
      setZoneId(res.zone_id);
      setCreatedZone({ zoneId: res.zone_id, name, description });
      setStatus("Zone created.");
    } catch (err) {
      setStatus(err.message);
    }
  }

  async function handleUpload() {
    try {
      const res = await uploadZoneKeyframes(zoneId, files);
      setStatus(`Uploaded ${res.count} keyframes.`);
    } catch (err) {
      setStatus(err.message);
    }
  }

  async function handleRebuild() {
    const res = await rebuildZoneIndex();
    setStatus(`Rebuilt zone index with ${res.count} items.`);
  }

  return (
    <div>
      <div className="page-title">Zone Manager</div>
      <p className="subtitle">
        Create simulated zones and upload keyframes for visual place recognition.
      </p>
      <div className="grid grid-3">
        <div className="card">
          <h3>Create Zone</h3>
          <div className="field">
            <label>Name</label>
            <input className="input" value={name} onChange={(e) => setName(e.target.value)} />
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
            Create Zone
          </button>
        </div>
        <div className="card">
          <h3>Upload Keyframes</h3>
          <div className="field">
            <label>Zone ID</label>
            <input className="input" value={zoneId} onChange={(e) => setZoneId(e.target.value)} />
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
          <p>Refresh FAISS with newly added keyframes.</p>
          <button className="button ghost" onClick={handleRebuild}>
            Rebuild
          </button>
        </div>
      </div>
      {createdZone && (
        <div style={{ marginTop: 20 }}>
          <ZoneCard {...createdZone} />
        </div>
      )}
      {status && <div className="mono" style={{ marginTop: 12 }}>{status}</div>}
    </div>
  );
}
