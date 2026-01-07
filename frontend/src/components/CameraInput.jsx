import { useEffect, useRef } from "react";

export default function CameraInput({ onFrame, onMediaSize, active, children }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const frameIndexRef = useRef(0);

  useEffect(() => {
    let stream;
    async function initCamera() {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    }
    initCamera();
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const onLoaded = () => {
      if (onMediaSize) {
        onMediaSize({ width: video.videoWidth, height: video.videoHeight });
      }
    };
    video.addEventListener("loadedmetadata", onLoaded);
    return () => video.removeEventListener("loadedmetadata", onLoaded);
  }, [onMediaSize]);

  useEffect(() => {
    if (!active) return;
    const interval = setInterval(() => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;
      const width = video.videoWidth;
      const height = video.videoHeight;
      if (!width || !height) return;
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, width, height);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
      onFrame?.(dataUrl, frameIndexRef.current);
      frameIndexRef.current += 1;
    }, 900);
    return () => clearInterval(interval);
  }, [active, onFrame]);

  return (
    <div className="media-stage">
      <video ref={videoRef} autoPlay playsInline muted />
      {children}
      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
}
