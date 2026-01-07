import { useEffect, useRef, useState } from "react";

export default function VideoInput({ onRun, onMediaSize, children }) {
  const [file, setFile] = useState(null);
  const videoRef = useRef(null);

  useEffect(() => {
    if (!videoRef.current) return;
    const handler = () => {
      if (onMediaSize) {
        onMediaSize({
          width: videoRef.current.videoWidth,
          height: videoRef.current.videoHeight,
        });
      }
    };
    videoRef.current.addEventListener("loadedmetadata", handler);
    return () => videoRef.current?.removeEventListener("loadedmetadata", handler);
  }, [onMediaSize]);

  function handleFileChange(event) {
    const selected = event.target.files[0];
    setFile(selected || null);
  }

  const videoUrl = file ? URL.createObjectURL(file) : null;

  useEffect(() => {
    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [videoUrl]);

  return (
    <div className="panel">
      <div className="field">
        <label>Upload video (mp4)</label>
        <input type="file" accept="video/*" onChange={handleFileChange} />
      </div>
      {file && (
        <div className="media-stage">
          <video ref={videoRef} src={videoUrl} controls />
          {children}
        </div>
      )}
      <div className="controls" style={{ marginTop: 12 }}>
        <button className="button" onClick={() => file && onRun(file)}>
          Run Inference
        </button>
      </div>
    </div>
  );
}
