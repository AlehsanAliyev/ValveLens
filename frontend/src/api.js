const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function handleResponse(res) {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Request failed");
  }
  return res.json();
}

export async function inferImage(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/infer/image`, {
    method: "POST",
    body: form,
  });
  return handleResponse(res);
}

export async function inferVideo(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/infer/video`, {
    method: "POST",
    body: form,
  });
  return handleResponse(res);
}

export async function inferWebcamFrame(payload) {
  const res = await fetch(`${API_BASE}/infer/webcam/frame`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse(res);
}

export async function createZone(data) {
  const res = await fetch(`${API_BASE}/zones/create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return handleResponse(res);
}

export async function uploadZoneKeyframes(zoneId, files) {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  const res = await fetch(`${API_BASE}/zones/${zoneId}/keyframes`, {
    method: "POST",
    body: form,
  });
  return handleResponse(res);
}

export async function rebuildZoneIndex() {
  const res = await fetch(`${API_BASE}/zones/rebuild_index`, {
    method: "POST",
  });
  return handleResponse(res);
}

export async function createDevice(data) {
  const res = await fetch(`${API_BASE}/devices/create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return handleResponse(res);
}

export async function uploadDeviceRefs(deviceId, files) {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  const res = await fetch(`${API_BASE}/devices/${deviceId}/refs`, {
    method: "POST",
    body: form,
  });
  return handleResponse(res);
}

export async function rebuildDeviceIndex() {
  const res = await fetch(`${API_BASE}/devices/rebuild_index`, {
    method: "POST",
  });
  return handleResponse(res);
}

export async function sendFeedback(payload) {
  const res = await fetch(`${API_BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse(res);
}

export async function getDebugStatus() {
  const res = await fetch(`${API_BASE}/debug/status`);
  return handleResponse(res);
}
