import { useNavigate } from "react-router-dom";

export default function Home() {
  const navigate = useNavigate();

  return (
    <div>
      <div className="page-title">ValveLens Demo</div>
      <p className="subtitle">
        Choose an input mode to run the vision pipeline. Webcam streams send frames
        live, while video and images are processed on demand.
      </p>
      <div className="grid grid-3">
        <div className="card">
          <h3>Webcam Live</h3>
          <p>Stream frames from your camera for interactive guidance.</p>
          <button className="button" onClick={() => navigate("/live/webcam")}>
            Open Webcam
          </button>
        </div>
        <div className="card">
          <h3>Video Upload</h3>
          <p>Process an mp4 frame-by-frame with configurable stride.</p>
          <button className="button secondary" onClick={() => navigate("/live/video")}>
            Upload Video
          </button>
        </div>
        <div className="card">
          <h3>Image Upload</h3>
          <p>Run a single frame through the full pipeline.</p>
          <button className="button ghost" onClick={() => navigate("/live/image")}>
            Upload Image
          </button>
        </div>
      </div>
    </div>
  );
}
