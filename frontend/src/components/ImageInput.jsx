import { useEffect, useRef, useState } from "react";

export default function ImageInput({ onRun, onMediaSize, children }) {
  const [file, setFile] = useState(null);
  const imgRef = useRef(null);

  useEffect(() => {
    if (!imgRef.current) return;
    const handler = () => {
      if (onMediaSize) {
        onMediaSize({
          width: imgRef.current.naturalWidth,
          height: imgRef.current.naturalHeight,
        });
      }
    };
    imgRef.current.addEventListener("load", handler);
    return () => imgRef.current?.removeEventListener("load", handler);
  }, [onMediaSize, file]);

  function handleFileChange(event) {
    const selected = event.target.files[0];
    setFile(selected || null);
  }

  const imageUrl = file ? URL.createObjectURL(file) : null;

  useEffect(() => {
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageUrl]);

  return (
    <div className="panel">
      <div className="field">
        <label>Upload image</label>
        <input type="file" accept="image/*" onChange={handleFileChange} />
      </div>
      {file && (
        <div className="media-stage">
          <img ref={imgRef} src={imageUrl} alt="Upload preview" />
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
