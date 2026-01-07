import { useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";

import {
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
  const sessionIdRef = useRef(
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : `session-${Date.now()}`
  );
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
      setStatus("");
    } catch (err) {
      setStatus(err.message);
    }
  }

  async function handleVideo(file) {
    try {
      const res = await inferVideo(file);
      setResponses(res);
      setCurrentIndex(0);
      setStatus("");
    } catch (err) {
      setStatus(err.message);
    }
  }

  async function handleTapSelect(det) {
    if (!activeResponse) return;
    await sendFeedback({
      obs_id: activeResponse.request_id,
      feedback_type: "tap_select",
      data_json: { det_id: det.det_id, device_id: det.fused.device_id },
    });
  }

  async function handleConfirm() {
    if (!activeResponse) return;
    await sendFeedback({
      obs_id: activeResponse.request_id,
      feedback_type: "confirm",
      data_json: { device_id: activeResponse.decision?.selected_device?.device_id },
    });
  }

  async function handleReject() {
    if (!activeResponse) return;
    await sendFeedback({
      obs_id: activeResponse.request_id,
      feedback_type: "reject",
      data_json: {},
    });
  }

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

      <SidePanel
        response={activeResponse}
        mode={interactionMode}
        onModeChange={setInteractionMode}
        onConfirm={handleConfirm}
        onReject={handleReject}
      />
    </div>
  );
}
