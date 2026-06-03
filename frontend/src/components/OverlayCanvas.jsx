import { useEffect, useRef } from "react";

function identityLabel(det, decision) {
  const classMissing = !det.class_name && !det.cls;
  if (classMissing) return "class name missing";
  const deviceId = det.fused?.device_id || decision?.selected_device?.device_id;
  if (decision?.status === "ACCEPTED" && deviceId) return `${deviceId} ACCEPTED`;
  if (decision?.status && decision.status !== "ACCEPTED") return "identity uncertain";
  if (deviceId) return deviceId;
  return "identity uncertain";
}

function getRenderedMediaRect(containerRect, mediaSize) {
  const mediaAspect = mediaSize.width / mediaSize.height;
  const containerAspect = containerRect.width / containerRect.height;

  if (containerAspect > mediaAspect) {
    const height = containerRect.height;
    const width = height * mediaAspect;
    return {
      x: (containerRect.width - width) / 2,
      y: 0,
      width,
      height,
    };
  }

  const width = containerRect.width;
  const height = width / mediaAspect;
  return {
    x: 0,
    y: (containerRect.height - height) / 2,
    width,
    height,
  };
}

function clampBBox(bbox, mediaSize) {
  if (!bbox) return null;
  let { x1, y1, x2, y2 } = bbox;
  x1 = Number(x1);
  y1 = Number(y1);
  x2 = Number(x2);
  y2 = Number(y2);
  if (![x1, y1, x2, y2].every(Number.isFinite)) return null;

  const appearsNormalized = Math.max(Math.abs(x1), Math.abs(y1), Math.abs(x2), Math.abs(y2)) <= 1.5;
  if (appearsNormalized) {
    x1 *= mediaSize.width;
    x2 *= mediaSize.width;
    y1 *= mediaSize.height;
    y2 *= mediaSize.height;
  }

  return {
    x1: Math.max(0, Math.min(mediaSize.width, x1)),
    y1: Math.max(0, Math.min(mediaSize.height, y1)),
    x2: Math.max(0, Math.min(mediaSize.width, x2)),
    y2: Math.max(0, Math.min(mediaSize.height, y2)),
  };
}

function drawDetections(ctx, detections, mediaSize, displayRect, decision) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.lineWidth = 2;
  ctx.font = "12px 'JetBrains Mono', monospace";

  detections.forEach((det) => {
    const bbox = clampBBox(det.bbox, mediaSize);
    if (!bbox) return;
    const scaleX = displayRect.width / mediaSize.width;
    const scaleY = displayRect.height / mediaSize.height;
    const left = displayRect.x + bbox.x1 * scaleX;
    const top = displayRect.y + bbox.y1 * scaleY;
    const width = (bbox.x2 - bbox.x1) * scaleX;
    const height = (bbox.y2 - bbox.y1) * scaleY;
    if (width <= 1 || height <= 1) return;

    ctx.strokeStyle = "#d3542e";
    ctx.fillStyle = "rgba(211, 84, 46, 0.15)";
    ctx.fillRect(left, top, width, height);
    ctx.strokeRect(left, top, width, height);
    const classLabel = det.class_name || det.cls || "unknown";
    const label = `${classLabel} ${(det.conf * 100).toFixed(0)}% | ${identityLabel(det, decision)}`;
    ctx.fillStyle = "#1e1b17";
    ctx.fillRect(left, Math.max(0, top - 18), ctx.measureText(label).width + 8, 16);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, left + 4, Math.max(12, top - 6));
  });
}

export default function OverlayCanvas({
  detections,
  mediaSize,
  decision = null,
  interactive = false,
  onSelect,
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    function render() {
      const canvas = canvasRef.current;
      if (!canvas || !mediaSize?.width || !mediaSize?.height) return;
      const parent = canvas.parentElement;
      if (!parent) return;
      const rect = parent.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      const displayRect = getRenderedMediaRect(rect, mediaSize);
      const ctx = canvas.getContext("2d");
      drawDetections(ctx, detections, mediaSize, displayRect, decision);
    }

    render();
    window.addEventListener("resize", render);
    return () => window.removeEventListener("resize", render);
  }, [detections, mediaSize, decision]);

  function handleClick(event) {
    if (!interactive || !onSelect || !mediaSize) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const displayRect = getRenderedMediaRect(rect, mediaSize);
    const canvasX = event.clientX - rect.left;
    const canvasY = event.clientY - rect.top;
    if (
      canvasX < displayRect.x ||
      canvasX > displayRect.x + displayRect.width ||
      canvasY < displayRect.y ||
      canvasY > displayRect.y + displayRect.height
    ) {
      return;
    }
    const x = ((canvasX - displayRect.x) / displayRect.width) * mediaSize.width;
    const y = ((canvasY - displayRect.y) / displayRect.height) * mediaSize.height;
    const hit = detections.find((det) => {
      const bbox = clampBBox(det.bbox, mediaSize);
      return bbox && x >= bbox.x1 && x <= bbox.x2 && y >= bbox.y1 && y <= bbox.y2;
    });
    if (hit) {
      onSelect(hit);
    }
  }

  return (
    <canvas
      ref={canvasRef}
      className="overlay-canvas"
      style={{ pointerEvents: interactive ? "auto" : "none" }}
      onClick={handleClick}
    />
  );
}
