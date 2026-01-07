import { useEffect, useRef } from "react";

function drawDetections(ctx, detections, scaleX, scaleY) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.lineWidth = 2;
  ctx.font = "12px 'JetBrains Mono', monospace";

  detections.forEach((det) => {
    const { x1, y1, x2, y2 } = det.bbox;
    const left = x1 * scaleX;
    const top = y1 * scaleY;
    const width = (x2 - x1) * scaleX;
    const height = (y2 - y1) * scaleY;
    ctx.strokeStyle = "#d3542e";
    ctx.fillStyle = "rgba(211, 84, 46, 0.15)";
    ctx.fillRect(left, top, width, height);
    ctx.strokeRect(left, top, width, height);
    const label = `${det.cls} ${(det.conf * 100).toFixed(0)}%`;
    ctx.fillStyle = "#1e1b17";
    ctx.fillRect(left, Math.max(0, top - 18), ctx.measureText(label).width + 8, 16);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, left + 4, Math.max(12, top - 6));
  });
}

export default function OverlayCanvas({
  detections,
  mediaSize,
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
      const scaleX = rect.width / mediaSize.width;
      const scaleY = rect.height / mediaSize.height;
      const ctx = canvas.getContext("2d");
      drawDetections(ctx, detections, scaleX, scaleY);
    }

    render();
    window.addEventListener("resize", render);
    return () => window.removeEventListener("resize", render);
  }, [detections, mediaSize]);

  function handleClick(event) {
    if (!interactive || !onSelect || !mediaSize) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = mediaSize.width / rect.width;
    const scaleY = mediaSize.height / rect.height;
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    const hit = detections.find(
      (det) => x >= det.bbox.x1 && x <= det.bbox.x2 && y >= det.bbox.y1 && y <= det.bbox.y2
    );
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
