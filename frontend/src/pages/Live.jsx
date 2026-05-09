import { useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";

import {
  askQuestion,
  getDebugStatus,
  inferImage,
  inferVideo,
  inferWebcamFrame,
  sendFeedback,
} from "../api";
import CameraInput from "../components/CameraInput";
import ImageInput from "../components/ImageInput";
import OverlayCanvas from "../components/OverlayCanvas";
import SidePanel from "../components/SidePanel";
import VideoInput from "../components/VideoInput";

export default function Live() {
  const { mode } = useParams();
  const [response, setResponse] = useState(null);
  const [responses, setResponses] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [mediaSize, setMediaSize] = useState(null);
  const [interactionMode, setInteractionMode] = useState("view");
  const [status, setStatus] = useState("");
  const [askResult, setAskResult] = useState(null);
  const [askStatus, setAskStatus] = useState("");
  const [selectedDetectionId, setSelectedDetectionId] = useState(null);
  const [debugStatus, setDebugStatus] = useState(null);
  const [debugError, setDebugError] = useState("");
  const sessionIdRef = useRef(
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : `session-${Date.now()}`
  );
  const videoSessionIdRef = useRef(null);
  const inFlightRef = useRef(false);

  const activeResponse = useMemo(() => {
    if (mode === "video") {
      return responses[currentIndex] || null;
    }
    return response;
  }, [mode, response, responses, currentIndex]);

  async function handleWebcamFrame(dataUrl, frameIndex) {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    try {
      const res = await inferWebcamFrame({
        session_id: sessionIdRef.current,
        frame_index: frameIndex,
        image_b64: dataUrl,
      });
      setResponse(res);
      setAskResult(null);
      setSelectedDetectionId(null);
      setStatus("");
    } catch (err) {
      setStatus(err.message);
    } finally {
      inFlightRef.current = false;
    }
  }

  async function handleImage(file) {
    try {
      const res = await inferImage(file);
      setResponse(res);
      setAskResult(null);
      setSelectedDetectionId(null);
      setStatus("");
    } catch (err) {
      setStatus(err.message);
    }
  }

  async function handleVideo(file) {
    try {
      const videoSessionId =
        typeof crypto !== "undefined" && crypto.randomUUID
          ? crypto.randomUUID()
          : `video-${Date.now()}`;
      videoSessionIdRef.current = videoSessionId;
      const res = await inferVideo(file, videoSessionId);
      setResponses(res);
      setCurrentIndex(0);
      setAskResult(null);
      setSelectedDetectionId(null);
      setStatus("");
    } catch (err) {
      setStatus(err.message);
    }
  }

  async function handleTapSelect(det) {
    if (!activeResponse) return;
    setSelectedDetectionId(det.det_id);
    const activeSessionId =
      mode === "webcam"
        ? sessionIdRef.current
        : mode === "video"
          ? videoSessionIdRef.current
          : null;
    const res = await sendFeedback({
      obs_id: activeResponse.request_id,
      feedback_type: "tap_select",
      data_json: { det_id: det.det_id, device_id: det.fused.device_id },
      session_id: activeSessionId,
    });
    if (res?.decision) {
      if (mode === "video") {
        setResponses((prev) => {
          const next = [...prev];
          next[currentIndex] = { ...next[currentIndex], decision: res.decision };
          return next;
        });
      } else {
        setResponse((prev) => ({ ...prev, decision: res.decision }));
      }
    }
  }

  async function handleConfirm() {
    if (!activeResponse) return;
    const activeSessionId =
      mode === "webcam"
        ? sessionIdRef.current
        : mode === "video"
          ? videoSessionIdRef.current
          : null;
    const res = await sendFeedback({
      obs_id: activeResponse.request_id,
      feedback_type: "confirm",
      data_json: { device_id: activeResponse.decision?.selected_device?.device_id },
      session_id: activeSessionId,
    });
    if (res?.decision) {
      if (mode === "video") {
        setResponses((prev) => {
          const next = [...prev];
          next[currentIndex] = { ...next[currentIndex], decision: res.decision };
          return next;
        });
      } else {
        setResponse((prev) => ({ ...prev, decision: res.decision }));
      }
    }
  }

  async function handleReject() {
    if (!activeResponse) return;
    const activeSessionId =
      mode === "webcam"
        ? sessionIdRef.current
        : mode === "video"
          ? videoSessionIdRef.current
          : null;
    const res = await sendFeedback({
      obs_id: activeResponse.request_id,
      feedback_type: "reject",
      data_json: {},
      session_id: activeSessionId,
    });
    if (res?.decision) {
      if (mode === "video") {
        setResponses((prev) => {
          const next = [...prev];
          next[currentIndex] = { ...next[currentIndex], decision: res.decision };
          return next;
        });
      } else {
        setResponse((prev) => ({ ...prev, decision: res.decision }));
      }
    }
  }

  async function handleAsk(question) {
    if (!activeResponse) return;
    const activeSessionId =
      mode === "webcam"
        ? sessionIdRef.current
        : mode === "video"
          ? videoSessionIdRef.current
          : null;
    try {
      setAskStatus("");
      const res = await askQuestion({
        question,
        session_id: activeSessionId,
        obs_id: activeResponse.request_id,
        selected_det_id: selectedDetectionId,
      });
      setAskResult(res);
    } catch (err) {
      setAskStatus(err.message);
    }
  }

  async function refreshDebugStatus() {
    try {
      const res = await getDebugStatus();
      setDebugStatus(res);
      setDebugError("");
    } catch (err) {
      setDebugError(err.message);
    }
  }

  useEffect(() => {
    refreshDebugStatus();
  }, []);

  const detections = activeResponse?.detections || [];

  return (
    <div className="live-layout">
      <div className="panel">
        {mode === "webcam" && (
          <CameraInput
            active
            onFrame={handleWebcamFrame}
            onMediaSize={setMediaSize}
          >
            <OverlayCanvas
              detections={detections}
              mediaSize={mediaSize}
              interactive={interactionMode === "tap"}
              onSelect={handleTapSelect}
            />
          </CameraInput>
        )}
        {mode === "video" && (
          <VideoInput onRun={handleVideo} onMediaSize={setMediaSize}>
            <OverlayCanvas
              detections={detections}
              mediaSize={mediaSize}
              interactive={interactionMode === "tap"}
              onSelect={handleTapSelect}
            />
          </VideoInput>
        )}
        {mode === "image" && (
          <ImageInput onRun={handleImage} onMediaSize={setMediaSize}>
            <OverlayCanvas
              detections={detections}
              mediaSize={mediaSize}
              interactive={interactionMode === "tap"}
              onSelect={handleTapSelect}
            />
          </ImageInput>
        )}
        {mode === "video" && responses.length > 0 && (
          <div className="field" style={{ marginTop: 12 }}>
            <label>Frame index</label>
            <input
              type="range"
              min="0"
              max={responses.length - 1}
              value={currentIndex}
              onChange={(e) => setCurrentIndex(Number(e.target.value))}
            />
          </div>
        )}
        {status && <div className="mono" style={{ marginTop: 8 }}>{status}</div>}
      </div>

      <div className="list">
        <SidePanel
          response={activeResponse}
          mode={interactionMode}
          onModeChange={setInteractionMode}
          onConfirm={handleConfirm}
          onReject={handleReject}
          onAsk={handleAsk}
          askResult={askResult}
          askStatus={askStatus}
        />
        <div className="panel">
          <div className="pill">System Status</div>
          <div className="controls" style={{ marginTop: 8 }}>
            <button className="button ghost" onClick={refreshDebugStatus}>
              Refresh
            </button>
          </div>
          {debugError && (
            <div className="mono" style={{ marginTop: 8 }}>{debugError}</div>
          )}
          {debugStatus && (
            <div className="list" style={{ marginTop: 10 }}>
              <div className="mono">
                zones: {debugStatus.counts?.zones ?? 0} | keyframes:{" "}
                {debugStatus.counts?.zone_keyframes ?? 0}
              </div>
              <div className="mono">
                devices: {debugStatus.counts?.devices ?? 0} | refs:{" "}
                {debugStatus.counts?.device_refs ?? 0}
              </div>
              <div className="mono">
                faiss zones: {debugStatus.faiss?.zones ?? 0} | faiss devices:{" "}
                {debugStatus.faiss?.devices ?? 0}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
